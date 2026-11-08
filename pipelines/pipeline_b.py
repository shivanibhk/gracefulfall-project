#Pipeline B: IMU + audio on MAC
#Second pipeline when PPG not available or 
#limited compute so PPG is dropped.

# The controller can dynamically select:
# B: IMU + audio
# C: IMU only
# F: audio only

import argparse
import asyncio
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from bleak import BleakClient, BleakScanner

from controller import (
    PipelineController,
    SensorHealth,
)
from detectors.audio_detector import (
    load_audio_model,
    predict_audio_samples,
)
from detectors.fusion_detector import (
    fuse_predictions,
)
from detectors.imu_detector import (
    predict_imu,
)
from fault_injection import (
    FaultAwareIMUCallback,
    FaultProfile,
)
from imu.imu_model import FallDetectorNN
from sensors.audio_stream import AudioStream
from sensors.imu_stream import IMUBuffer

CHAR_UUID = (
    "00002A56-0000-1000-8000-00805f9b34fb"
)


AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SECONDS = 3.0

AUDIO_WINDOW_SAMPLES = int(
    AUDIO_SAMPLE_RATE
    * AUDIO_WINDOW_SECONDS
)

IMU_TARGET_RATE = 50
IMU_WINDOW_SECONDS = 3.0

IMU_WINDOW_SAMPLES = int(
    IMU_TARGET_RATE
    * IMU_WINDOW_SECONDS
)

# The current SisFall model was trained with:
# 100 samples x 6 IMU channels.
#
# At 50 Hz, 100 samples represents approximately
# 2 seconds. The buffer is still 3 seconds so that
# future 3-second IMU models can be used.
IMU_MODEL_SAMPLES = 100


FALL_THRESHOLD = 0.80

REQUIRED_HIGH_PREDICTIONS = 1

ALERT_COOLDOWN_SECONDS = 5.0


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.file = Path(path).open(
            "w",
            newline="",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fieldnames,
        )

        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def write_imu_sample(logger, row):
    logger.write({
        "host_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        **row,
    })


def create_loggers(output_prefix):
    imu_logger = CSVLogger(
        f"{output_prefix}_imu_samples.csv",
        [
            "host_timestamp",
            "device_timestamp_ms",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ],
    )

    inference_logger = CSVLogger(
        f"{output_prefix}_inference.csv",
        [
            "host_timestamp",
            "pipeline",
            "pipeline_switched",
            "switch_reason",
            "window_number",

            "imu_available",
            "audio_available",
            "imu_rate_hz",
            "audio_rate_hz",

            "imu_sample_count",
            "audio_samples",
            "audio_frames",
            "audio_dropped_frames",

            "imu_normal_probability",
            "imu_fall_probability",

            "audio_normal_probability",
            "audio_fall_probability",

            "fused_fall_probability",

            "free_fall",
            "impact",
            "orientation_change",
            "post_event_quiet",
            "fall_signature",

            "decision",
            "decision_probability",
            "alert",
        ],
    )

    return imu_logger, inference_logger


class AlertPolicy:
    def __init__(self):
        self.high_prediction_count = 0
        self.last_alert_time = -float("inf")

    def update(
        self,
        fall_probability,
        motion=None,
    ):
        """
        For Pipeline B or C:

            high model probability
            AND IMU motion signature

        For Pipeline F:

            high audio probability

        Pipeline F has no IMU motion result, so motion
        is None and only the audio probability is used.
        """

        candidate = (
            fall_probability >= FALL_THRESHOLD
        )

        if motion is not None:
            candidate = (
                candidate
                and motion["fall_signature"]
            )

        if candidate:
            self.high_prediction_count += 1
        else:
            self.high_prediction_count = 0

        now = time.monotonic()

        cooldown_complete = (
            now - self.last_alert_time
            >= ALERT_COOLDOWN_SECONDS
        )

        alert = (
            self.high_prediction_count
            >= REQUIRED_HIGH_PREDICTIONS
            and cooldown_complete
        )

        if alert:
            self.last_alert_time = now
            self.high_prediction_count = 0

        return alert


def load_imu_model(path):
    model = FallDetectorNN()

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state" in checkpoint
    ):
        checkpoint = checkpoint[
            "model_state"
        ]

    model.load_state_dict(checkpoint)
    model.eval()

    return model


async def find_nano():
    print("Scanning for NanoBLE...")

    discovered = await BleakScanner.discover(
        timeout=15,
        return_adv=True,
    )

    for _, result in discovered.items():
        device, advertisement = result

        device_name = device.name or ""
        local_name = (
            advertisement.local_name
            or ""
        )

        if (
            device_name == "NanoBLE"
            or local_name == "NanoBLE"
        ):
            return device

    return None



def format_empty(result, key):
    if result is None:
        return ""

    return result[key]


async def run_pipeline(
    port,
    duration,
    output_prefix,
    audio_enabled=True,
    imu_enabled=True,
    audio_drop_every=0,
    imu_drop_every=0,
    audio_delay_ms=0.0,
    imu_delay_ms=0.0,
    inference_delay_ms=0.0,
):
    print("Loading IMU model...")

    imu_model = load_imu_model(
        "imu/sisfall_heavy_model.pth"
    )

    print("Loading audio model...")

    audio_model = load_audio_model(
        "audio_fall_detection/audio_model.pth"
    )

    fault_profile = FaultProfile(
        audio_enabled=audio_enabled,
        imu_enabled=imu_enabled,
        audio_drop_every=audio_drop_every,
        imu_drop_every=imu_drop_every,
        audio_delay_ms=audio_delay_ms,
        imu_delay_ms=imu_delay_ms,
        inference_delay_ms=inference_delay_ms,
    )

    controller = PipelineController()

    imu_logger, inference_logger = (
        create_loggers(output_prefix)
    )

    imu_buffer = IMUBuffer(
        window_size=IMU_WINDOW_SAMPLES,
        sample_logger=lambda row: (
            write_imu_sample(
                imu_logger,
                row,
            )
        ),
    )

    imu_callback = FaultAwareIMUCallback(
        imu_buffer=imu_buffer,
        fault_profile=fault_profile,
    )

    audio_stream = AudioStream(
        port=port,
        sample_rate=AUDIO_SAMPLE_RATE,
        window_seconds=AUDIO_WINDOW_SECONDS,
        frame_filter=(
            fault_profile.keep_audio_frame
        ),
        frame_delay=(
            fault_profile.delay_audio
        ),
    )

    alert_policy = AlertPolicy()

    try:
        print("Opening USB audio connection...")

        # Opens the serial port but does not send
        # sensor data yet.
        audio_stream.open()

        nano = await find_nano()

        if nano is None:
            print("Could not find NanoBLE.")
            return

        print(
            f"Found NanoBLE at {nano.address}."
        )

        async with BleakClient(
            nano.address
        ) as client:
            print(
                "Connected to NanoBLE."
            )

            await client.start_notify(
                CHAR_UUID,
                imu_callback,
            )

            print(
                "Starting IMU and audio "
                "streams together..."
            )

            # Sends START to the Nano.
            # The Nano then begins both IMU
            # and audio transmission.
            audio_stream.start_streaming()

            print(
                "IMU and audio streaming started."
            )

            session_start = time.monotonic()
            last_inference_time = -float("inf")
            window_number = 0

            try:
                while (
                    time.monotonic()
                    - session_start
                    < duration
                ):
                    await asyncio.sleep(0.05)

                    #Measure current sensor health
                    audio_stats = (
                        audio_stream.stats()
                    )

                    elapsed = max(
                        time.monotonic()
                        - session_start,
                        1e-6,
                    )

                    imu_rate = (
                        imu_buffer.measured_rate_hz()
                    )

                    audio_rate = (
                        audio_stats[
                            "samples_received"
                        ]
                        / elapsed
                    )

                    health = SensorHealth(
                        imu_available=(
                            imu_enabled
                            and imu_rate >= 30.0
                        ),
                        audio_available=(
                            audio_enabled
                            and audio_rate >= 8000.0
                        ),
                        ppg_available=False,
                        imu_rate_hz=imu_rate,
                        audio_rate_hz=audio_rate,
                        ble_latency_ms=0.0,
                        cpu_load=0.0,
                        mac_available=True,
                        nano_model_available=False,
                    )

                    #controller picks B, C, F
                    controller_result = (
                        controller.update(
                            health
                        )
                    )

                    selected_pipeline = (
                        controller_result[
                            "pipeline"
                        ]
                    )

                    if controller_result[
                        "switched"
                    ]:
                        print(
                            f"\nPipeline switched: "
                            f"{controller_result['reason']}"
                        )

                    #Only check sensors required by pipeline
                    imu_ready = (
                        imu_buffer.ready()
                    )

                    audio_ready = (
                        audio_stream.ready()
                    )

                    if (
                        selected_pipeline
                        == "B"
                    ):
                        if (
                            not imu_ready
                            or not audio_ready
                        ):
                            continue

                    elif (
                        selected_pipeline
                        == "C"
                    ):
                        if not imu_ready:
                            continue

                    elif (
                        selected_pipeline
                        == "F"
                    ):
                        if not audio_ready:
                            continue

                    else:
                        print(
                            f"\rNo usable pipeline: "
                            f"{selected_pipeline}",
                            end="",
                            flush=True,
                        )
                        continue

                    # Run an inference on every three-second interval
                    current_time = time.monotonic()

                    if (
                        current_time
                        - last_inference_time
                        < AUDIO_WINDOW_SECONDS
                    ):
                        continue

                    last_inference_time = (
                        current_time
                    )

                    window_number += 1

                    fault_profile.delay_inference()

                    imu_result = None
                    audio_result = None
                    fusion_result = None
                    motion = None

                    
                    if (
                        selected_pipeline
                        == "B"
                    ):
                        imu_window_3s = (
                            imu_buffer.snapshot()
                        )

                        audio_window_3s = (
                            audio_stream.snapshot()
                        )

                        if (
                            imu_window_3s is None
                            or audio_window_3s is None
                        ):
                            continue

                        # Current IMU model expects 100 samples
                        # Approximately two seconds at 50 Hz
                        imu_window_2s = (
                            imu_window_3s[
                                -IMU_MODEL_SAMPLES:
                            ]
                        )
                        imu_result = predict_imu(
                            model=imu_model,
                            window=imu_window_2s,
                        )
                        audio_result = (
                            predict_audio_samples(
                                model=audio_model,
                                samples=audio_window_3s,
                            )
                        )

                        fusion_result = (
                            fuse_predictions(
                                imu_result=imu_result,
                                audio_result=audio_result,
                            )
                        )

                        decision_probability = (
                            fusion_result[
                                "fall_probability"
                            ]
                        )
                        decision = (
                            fusion_result[
                                "decision"
                            ]
                        )

                        motion = imu_result[
                            "motion"
                        ]


                    elif (
                        selected_pipeline
                        == "C"
                    ):
                        imu_window_3s = (
                            imu_buffer.snapshot()
                        )
                        if imu_window_3s is None:
                            continue
                        
                        imu_window_2s = (
                            imu_window_3s[
                                -IMU_MODEL_SAMPLES:
                            ]
                        )
                        imu_result = predict_imu(
                            model=imu_model,
                            window=imu_window_2s,
                        )

                        decision_probability = (
                            imu_result[
                                "fall_probability"
                            ]
                        )
                        decision = (
                            "fall"
                            if decision_probability
                            >= FALL_THRESHOLD
                            else "normal"
                        )

                        motion = imu_result[
                            "motion"
                        ]

                    elif (
                        selected_pipeline
                        == "F"
                    ):
                        audio_window_3s = (
                            audio_stream.snapshot()
                        )
                        if audio_window_3s is None:
                            continue

                        audio_result = (
                            predict_audio_samples(
                                model=audio_model,
                                samples=audio_window_3s,
                            )
                        )

                        decision_probability = (
                            audio_result[
                                "fall_probability"
                            ]
                        )
                        decision = (
                            "fall"
                            if decision_probability
                            >= FALL_THRESHOLD
                            else "normal"
                        )

                    else:
                        continue

                    alert = alert_policy.update(
                        fall_probability=(
                            decision_probability
                        ),
                        motion=motion,
                    )

                    #print live result
                    imu_probability = (
                        ""
                        if imu_result is None
                        else (
                            f"{imu_result['fall_probability']:.3f}"
                        )
                    )

                    audio_probability = (
                        ""
                        if audio_result is None
                        else (
                            f"{audio_result['fall_probability']:.3f}"
                        )
                    )

                    fused_probability = (
                        ""
                        if fusion_result is None
                        else (
                            f"{fusion_result['fall_probability']:.3f}"
                        )
                    )

                    impact = (
                        ""
                        if motion is None
                        else str(
                            motion["impact"]
                        )
                    )

                    print(
                        f"\rWindow "
                        f"{window_number:04d} | "
                        f"pipeline="
                        f"{selected_pipeline} | "
                        f"P(IMU)="
                        f"{imu_probability} | "
                        f"P(audio)="
                        f"{audio_probability} | "
                        f"P(fused)="
                        f"{fused_probability} | "
                        f"IMU rate="
                        f"{imu_rate:.1f} Hz | "
                        f"audio rate="
                        f"{audio_rate:.0f} Hz | "
                        f"impact="
                        f"{impact} | "
                        f"decision="
                        f"{decision} | "
                        f"alert="
                        f"{alert}",
                        end="",
                        flush=True,
                    )

                    #CSV logging
                    inference_logger.write({
                        "host_timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),

                        "pipeline": (
                            selected_pipeline
                        ),

                        "pipeline_switched": (
                            controller_result[
                                "switched"
                            ]
                        ),

                        "switch_reason": (
                            controller_result[
                                "reason"
                            ]
                        ),

                        "window_number": (
                            window_number
                        ),

                        "imu_available": (
                            health.imu_available
                        ),

                        "audio_available": (
                            health.audio_available
                        ),

                        "imu_rate_hz": (
                            imu_rate
                        ),

                        "audio_rate_hz": (
                            audio_rate
                        ),

                        "imu_sample_count": (
                            imu_buffer.stats.sample_count
                        ),

                        "audio_samples": (
                            audio_stats[
                                "samples_received"
                            ]
                        ),

                        "audio_frames": (
                            audio_stats[
                                "frames_received"
                            ]
                        ),

                        "audio_dropped_frames": (
                            audio_stats.get(
                                "dropped_frames",
                                0,
                            )
                        ),

                        "imu_normal_probability": (
                            format_empty(
                                imu_result,
                                "normal_probability",
                            )
                        ),

                        "imu_fall_probability": (
                            format_empty(
                                imu_result,
                                "fall_probability",
                            )
                        ),

                        "audio_normal_probability": (
                            format_empty(
                                audio_result,
                                "normal_probability",
                            )
                        ),

                        "audio_fall_probability": (
                            format_empty(
                                audio_result,
                                "fall_probability",
                            )
                        ),

                        "fused_fall_probability": (
                            format_empty(
                                fusion_result,
                                "fall_probability",
                            )
                        ),

                        "free_fall": (
                            format_empty(
                                motion,
                                "free_fall",
                            )
                        ),

                        "impact": (
                            format_empty(
                                motion,
                                "impact",
                            )
                        ),

                        "orientation_change": (
                            format_empty(
                                motion,
                                "orientation_change",
                            )
                        ),

                        "post_event_quiet": (
                            format_empty(
                                motion,
                                "post_event_quiet",
                            )
                        ),

                        "fall_signature": (
                            format_empty(
                                motion,
                                "fall_signature",
                            )
                        ),

                        "decision": decision,

                        "decision_probability": (
                            decision_probability
                        ),

                        "alert": alert,
                    })

                    if alert:
                        print(
                            "\n\n*** FALL DETECTED BY "
                            f"{selected_pipeline} ***\n"
                        )

            finally:
                await client.stop_notify(
                    CHAR_UUID
                )

    finally:
        audio_stream.stop()
        imu_logger.close()
        inference_logger.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        required=True,
        help="Nano USB serial port",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Runtime in seconds",
    )

    parser.add_argument(
        "--output-prefix",
        default="pipeline_b_live",
    )

    parser.add_argument(
        "--disable-audio",
        action="store_true",
    )

    parser.add_argument(
        "--disable-imu",
        action="store_true",
    )

    parser.add_argument(
        "--audio-drop-every",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--imu-drop-every",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--audio-delay-ms",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--imu-delay-ms",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--inference-delay-ms",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            port=args.port,
            duration=args.duration,
            output_prefix=args.output_prefix,
            audio_enabled=(
                not args.disable_audio
            ),
            imu_enabled=(
                not args.disable_imu
            ),
            audio_drop_every=(
                args.audio_drop_every
            ),
            imu_drop_every=(
                args.imu_drop_every
            ),
            audio_delay_ms=(
                args.audio_delay_ms
            ),
            imu_delay_ms=(
                args.imu_delay_ms
            ),
            inference_delay_ms=(
                args.inference_delay_ms
            ),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline B stopped.")