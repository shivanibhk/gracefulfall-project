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
    predict_audio,
)
from detectors.fusion_detector import (
    fuse_predictions,
)
from detectors.imu_detector import predict_imu
from imu.imu_model import FallDetectorNN
from sensors.imu_stream import IMUBuffer


CHAR_UUID = (
    "00002A56-0000-1000-8000-00805f9b34fb"
)

WINDOW_SIZE = 100
INFERENCE_EVERY_N_SAMPLES = 5

FALL_THRESHOLD = 0.80
REQUIRED_HIGH_PREDICTIONS = 2
ALERT_COOLDOWN_SECONDS = 5


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)

        self.file = self.path.open(
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
    def __init__(
        self,
        threshold=FALL_THRESHOLD,
        required_predictions=(
            REQUIRED_HIGH_PREDICTIONS
        ),
        cooldown_seconds=(
            ALERT_COOLDOWN_SECONDS
        ),
    ):
        self.threshold = threshold
        self.required_predictions = (
            required_predictions
        )
        self.cooldown_seconds = (
            cooldown_seconds
        )

        self.high_prediction_count = 0
        self.last_alert_time = -float("inf")

    def update(self, fall_probability):
        if fall_probability >= self.threshold:
            self.high_prediction_count += 1
        else:
            self.high_prediction_count = 0

        current_time = time.monotonic()

        cooldown_complete = (
            current_time - self.last_alert_time
            >= self.cooldown_seconds
        )

        should_alert = (
            self.high_prediction_count
            >= self.required_predictions
            and cooldown_complete
        )

        if should_alert:
            self.last_alert_time = current_time
            self.high_prediction_count = 0

        return should_alert


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


def load_imu_model(model_path):
    model = FallDetectorNN()

    checkpoint = torch.load(
        model_path,
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


def create_loggers(output_prefix):
    imu_logger = CSVLogger(
        f"{output_prefix}_imu_samples.csv",
        [
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
            "inference_number",
            "imu_sample_count",
            "imu_rate_hz",
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


async def run_pipeline(
    audio_path,
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

    print(
        f"Using audio file: {audio_path}"
    )

    audio_result = predict_audio(
        audio_model,
        audio_path,
    )
    # Keep audio usable for now.
    # The current audio score participates in fusion.
    audio_result["usable"] = True

    print(
        "Audio prediction: "
        f"P(normal)="
        f"{audio_result['normal_probability']:.3f} | "
        f"P(fall)="
        f"{audio_result['fall_probability']:.3f}"
    )

    print(
        "NOTE: this audio score is currently "
        "computed once from the WAV file and "
        "reused for each IMU window."
    )

    imu_logger, inference_logger = (
        create_loggers(output_prefix)
    )

    imu_buffer = IMUBuffer(
        window_size=WINDOW_SIZE,
        sample_logger=imu_logger.write,
    )

    alert_policy = AlertPolicy()

    nano = await find_nano()

    if nano is None:
        imu_logger.close()
        inference_logger.close()

        print("Could not find NanoBLE.")
        return

    print(
        f"Found NanoBLE at {nano.address}."
    )

    async with BleakClient(nano.address) as client:
        print(
            "Connected. Starting "
            "IMU/audio fusion."
        )

        await client.start_notify(
            CHAR_UUID,
            imu_buffer.parse_packet,
        )

        session_start = time.monotonic()
        last_inference_sample = 0
        inference_number = 0

        try:
            while True:
                elapsed = (
                    time.monotonic()
                    - session_start
                )

                if elapsed >= duration:
                    print()
                    print(
                        f"Completed {duration:.2f}-second "
                        "Pipeline B test."
                    )
                    break

                await asyncio.sleep(0.05)

                if not imu_buffer.ready():
                    print(
                        "\rCollecting IMU samples: "
                        f"{len(imu_buffer.buffer)}/"
                        f"{WINDOW_SIZE}",
                        end="",
                        flush=True,
                    )
                    continue

                current_sample_count = (
                    imu_buffer.stats.sample_count
                )

                if (
                    current_sample_count
                    - last_inference_sample
                    < INFERENCE_EVERY_N_SAMPLES
                ):
                    continue

                last_inference_sample = (
                    current_sample_count
                )

                inference_number += 1

                window = imu_buffer.snapshot()

                if window is None:
                    continue

                imu_result = predict_imu(
                    model=imu_model,
                    window=window,
                )

                fusion_result = fuse_predictions(
                    imu_result=imu_result,
                    audio_result=audio_result,
                )

                alert = alert_policy.update(
                    fusion_result[
                        "fall_probability"
                    ]
                )

                motion = imu_result["motion"]

                print(
                    f"\rInference "
                    f"{inference_number:04d} | "
                    f"P(IMU fall)="
                    f"{imu_result['fall_probability']:.3f} | "
                    f"P(audio fall)="
                    f"{audio_result['fall_probability']:.3f} | "
                    f"P(combined fall)="
                    f"{fusion_result['fall_probability']:.3f} | "
                    f"impact="
                    f"{motion['impact']}",
                    end="",
                    flush=True,
                )

                inference_logger.write({
                    "host_timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "pipeline": "B_imu_audio_mac",

                    "inference_number": (
                        inference_number
                    ),

                    "imu_sample_count": (
                        imu_buffer.stats.sample_count
                    ),

                    "imu_rate_hz": (
                        imu_buffer.measured_rate_hz()
                    ),

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
                    print()
                    print(
                        "\n*** FALL DETECTED "
                        "BY PIPELINE B ***\n"
                    )

        finally:
            await client.stop_notify(
                CHAR_UUID
            )

            imu_logger.close()
            inference_logger.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        required=True,
        help="Audio WAV file",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help=(
            "How long Pipeline B should run "
            "after BLE connection"
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="pipeline_b_test",
        help="Prefix for CSV files",
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            audio_path=args.audio,
            duration=args.duration,
            output_prefix=args.output_prefix,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline B stopped.")