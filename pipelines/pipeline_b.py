#Pipeline B: IMU + audio on MAC
#Second pipeline when PPG not available or 
#limited compute so PPG is dropped.

import argparse
import asyncio
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from bleak import BleakClient, BleakScanner

from detectors.audio_detector import (
    load_audio_model,
    predict_audio_samples,
)
from detectors.fusion_detector import (
    fuse_predictions,
)
from detectors.imu_detector import predict_imu
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

# The current SisFall model expects:
# 100 samples x 6 channels
IMU_MODEL_SAMPLES = 100

# Process one new multimodal window
# every approximately three seconds.
WINDOW_STRIDE_SAMPLES = IMU_WINDOW_SAMPLES

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


class AlertPolicy:
    def __init__(self):
        self.high_prediction_count = 0
        self.last_alert_time = -float("inf")

    def update(self, fusion_result, imu_result):
        motion = imu_result["motion"]

        candidate = (
            fusion_result["fall_probability"]
            >= FALL_THRESHOLD
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
            "window_number",
            "imu_sample_count",
            "imu_rate_hz",
            "audio_samples",
            "audio_frames",
            "audio_rate_hz",
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
            "alert",
        ],
    )

    return imu_logger, inference_logger


def write_imu_sample(
    logger,
    row,
):
    logger.write({
        "host_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        **row,
    })


async def run_pipeline(
    port,
    duration,
    output_prefix,
):
    print("Loading IMU model...")

    imu_model = load_imu_model(
        "imu/sisfall_heavy_model.pth"
    )

    print("Loading audio model...")

    audio_model = load_audio_model(
        "audio_fall_detection/audio_model.pth"
    )

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

    audio_stream = AudioStream(
        port=port,
        sample_rate=AUDIO_SAMPLE_RATE,
        window_seconds=AUDIO_WINDOW_SECONDS,
    )

    alert_policy = AlertPolicy()

    try:
        print("Opening USB audio connection...")

        # Opens the port but does not send
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
                imu_buffer.parse_packet,
            )

            print(
                "Starting IMU and audio "
                "streams together..."
            )

            # This sends START to the Nano.
            # The Nano then begins both IMU
            # and audio transmission.
            audio_stream.start_streaming()

            print(
                "IMU and audio streaming started."
            )

            session_start = time.monotonic()
            last_window_sample = 0
            window_number = 0

            try:
                while (
                    time.monotonic()
                    - session_start
                    < duration
                ):
                    await asyncio.sleep(0.05)

                    audio_stats = (
                        audio_stream.stats()
                    )

                    if not imu_buffer.ready():
                        print(
                            "\rWaiting for IMU: "
                            f"{len(imu_buffer.buffer)}/"
                            f"{IMU_WINDOW_SAMPLES}",
                            end="",
                            flush=True,
                        )
                        continue

                    if not audio_stream.ready():
                        print(
                            "\rWaiting for audio: "
                            f"{audio_stats['buffer_samples']}/"
                            f"{AUDIO_WINDOW_SAMPLES}",
                            end="",
                            flush=True,
                        )
                        continue

                    current_imu_count = (
                        imu_buffer.stats.sample_count
                    )

                    if (
                        current_imu_count
                        - last_window_sample
                        < WINDOW_STRIDE_SAMPLES
                    ):
                        continue

                    last_window_sample = (
                        current_imu_count
                    )
                    window_number += 1

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

                    # The current SisFall model accepts
                    # 100 samples, approximately two
                    # seconds at 50 Hz.
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

                    # Keep audio active.
                    audio_result["usable"] = True

                    fusion_result = (
                        fuse_predictions(
                            imu_result=imu_result,
                            audio_result=audio_result,
                        )
                    )

                    alert = alert_policy.update(
                        fusion_result=fusion_result,
                        imu_result=imu_result,
                    )

                    motion = imu_result["motion"]

                    elapsed = (
                        time.monotonic()
                        - session_start
                    )

                    if elapsed > 0:
                        audio_rate = (
                            audio_stats[
                                "samples_received"
                            ]
                            / elapsed
                        )
                    else:
                        audio_rate = 0.0

                    print(
                        f"\rWindow {window_number:04d} | "
                        f"P(IMU)= {imu_result['fall_probability']:.3f} | "
                        f"P(audio)= {audio_result['fall_probability']:.3f} | "
                        f"P(fused)= {fusion_result['fall_probability']:.3f} | "
                        f"IMU rate= {imu_buffer.measured_rate_hz():.1f} Hz | "
                        f"audio rate= {audio_rate:.0f} Hz | "
                        f"impact= {motion['impact']}",
                        end="",
                        flush=True,
                    )

                    inference_logger.write({
                        "host_timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),

                        "pipeline": (
                            "B_imu_audio_mac"
                        ),

                        "window_number": (
                            window_number
                        ),

                        "imu_sample_count": (
                            current_imu_count
                        ),

                        "imu_rate_hz": (
                            imu_buffer.measured_rate_hz()
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

                        "audio_rate_hz": audio_rate,

                        "imu_normal_probability": (
                            imu_result[
                                "normal_probability"
                            ]
                        ),

                        "imu_fall_probability": (
                            imu_result[
                                "fall_probability"
                            ]
                        ),

                        "audio_normal_probability": (
                            audio_result[
                                "normal_probability"
                            ]
                        ),

                        "audio_fall_probability": (
                            audio_result[
                                "fall_probability"
                            ]
                        ),

                        "fused_fall_probability": (
                            fusion_result[
                                "fall_probability"
                            ]
                        ),

                        "free_fall": (
                            motion["free_fall"]
                        ),

                        "impact": (
                            motion["impact"]
                        ),

                        "orientation_change": (
                            motion[
                                "orientation_change"
                            ]
                        ),

                        "post_event_quiet": (
                            motion[
                                "post_event_quiet"
                            ]
                        ),

                        "fall_signature": (
                            motion[
                                "fall_signature"
                            ]
                        ),

                        "decision": (
                            fusion_result[
                                "decision"
                            ]
                        ),

                        "alert": alert,
                    })

                    if alert:
                        print(
                            "\n\n*** FALL DETECTED "
                            "BY PIPELINE B ***\n"
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

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            port=args.port,
            duration=args.duration,
            output_prefix=args.output_prefix,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline B stopped.")