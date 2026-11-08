#Pipeline F: audio on MAC
#Worst-case, emergency pipeline if IMU not available.
import argparse
import asyncio
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from detectors.audio_detector import (
    load_audio_model,
    predict_audio_samples,
)
from sensors.audio_stream import AudioStream

SAMPLE_RATE = 16000
DEFAULT_WINDOW_SECONDS = 3.0
INFERENCE_EVERY_SECONDS = 0.5

FALL_THRESHOLD = 0.80
REQUIRED_HIGH_PREDICTIONS = 2
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


class AudioAlertPolicy:
    def __init__(self):
        self.high_prediction_count = 0
        self.last_alert_time = -float("inf")

    def update(self, fall_probability):
        if fall_probability >= FALL_THRESHOLD:
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


def create_logger(output_path):
    return CSVLogger(
        output_path,
        [
            "host_timestamp",
            "pipeline",
            "inference_number",
            "audio_samples",
            "audio_frames",
            "audio_rate_hz",
            "normal_probability",
            "fall_probability",
            "decision",
            "alert",
        ],
    )


async def run_pipeline(
    port,
    duration,
    model_path,
    output_path,
    window_seconds,
):
    print("Loading audio model...")
    audio_model = load_audio_model(model_path)

    audio_stream = AudioStream(
        port=port,
        sample_rate=SAMPLE_RATE,
        window_seconds=window_seconds,
    )

    logger = create_logger(output_path)
    alert_policy = AudioAlertPolicy()

    try:
        print(
            f"Starting audio-only stream with a "
            f"{window_seconds:.1f}-second window..."
        )
        audio_stream.start()

        session_start = time.monotonic()
        next_inference_time = session_start
        inference_number = 0

        while (
            time.monotonic() - session_start
            < duration
        ):
            await asyncio.sleep(0.05)

            stats = audio_stream.stats()

            if not audio_stream.ready():
                print(
                    f"\rWaiting for audio: "
                    f"{stats['buffer_samples']}/"
                    f"{audio_stream.window_size} samples",
                    end="",
                    flush=True,
                )
                continue

            now = time.monotonic()
            if now < next_inference_time:
                continue

            next_inference_time = now + INFERENCE_EVERY_SECONDS
            inference_number += 1

            audio_window = audio_stream.snapshot()
            if audio_window is None:
                continue

            audio_result = predict_audio_samples(
                model=audio_model,
                samples=audio_window,
            )

            fall_probability = audio_result[
                "fall_probability"
            ]
            decision = (
                "fall"
                if fall_probability >= FALL_THRESHOLD
                else "normal"
            )
            alert = alert_policy.update(
                fall_probability
            )

            elapsed = max(
                time.monotonic() - session_start,
                1e-6,
            )
            audio_rate = (
                stats["samples_received"] / elapsed
            )

            print(
                f"\rInference {inference_number:04d} | "
                f"P(audio fall)="
                f"{fall_probability:.3f} | "
                f"rate={audio_rate:.1f} Hz | "
                f"decision={decision} | "
                f"alert={alert}",
                end="",
                flush=True,
            )

            logger.write({
                "host_timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "pipeline": "F_audio_mac",
                "inference_number": inference_number,
                "audio_samples": stats[
                    "samples_received"
                ],
                "audio_frames": stats[
                    "frames_received"
                ],
                "audio_rate_hz": audio_rate,
                "normal_probability": audio_result[
                    "normal_probability"
                ],
                "fall_probability": fall_probability,
                "decision": decision,
                "alert": alert,
            })

            if alert:
                print(
                    "\n\n*** FALL DETECTED "
                    "BY PIPELINE F ***\n"
                )

    finally:
        audio_stream.stop()
        logger.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run audio-only Pipeline F."
    )

    parser.add_argument(
        "--port",
        required=True,
        help="Nano USB serial port for audio",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--model",
        default="audio_fall_detection/audio_model.pth",
    )
    parser.add_argument(
        "--output",
        default="pipeline_f_live.csv",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Must match the audio training clip length.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            port=args.port,
            duration=args.duration,
            model_path=args.model,
            output_path=args.output,
            window_seconds=args.window_seconds,
        )
    )


if __name__ == "__main__":
    main()
