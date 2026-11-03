import asyncio
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from bleak import BleakClient, BleakScanner


class FallDetectorNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 25, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # Input shape:
        # batch × samples × channels
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x


CHAR_UUID = (
    "00002A56-0000-1000-8000-00805f9b34fb"
)

WINDOW_SIZE = 100
INFERENCE_EVERY_N_SAMPLES = 5

FALL_PROBABILITY_THRESHOLD = 0.80
REQUIRED_HIGH_PREDICTIONS = 2
ALERT_COOLDOWN_SECONDS = 5


window_buffer = deque(maxlen=WINDOW_SIZE)

sample_count = 0
first_device_timestamp = None
last_device_timestamp = None


def parse_packet(sender, data):
    global sample_count
    global first_device_timestamp
    global last_device_timestamp

    try:
        decoded = data.decode("utf-8").strip()
        parts = decoded.split(",")

        # timestamp, ax, ay, az, gx, gy, gz
        if len(parts) != 7:
            print(f"Ignoring packet with {len(parts)} fields:")
            print(decoded)
            return

        device_timestamp = int(parts[0])

        features = [
            float(value)
            for value in parts[1:]
        ]

        if first_device_timestamp is None:
            first_device_timestamp = device_timestamp

        last_device_timestamp = device_timestamp

        window_buffer.append(features)
        sample_count += 1

    except (UnicodeDecodeError, ValueError) as error:
        print(f"Ignoring malformed packet: {error}")


async def find_nano():
    print("Scanning for NanoBLE...")

    discovered = await BleakScanner.discover(
        timeout=15,
        return_adv=True,
    )

    for address, result in discovered.items():
        device, advertisement = result

        device_name = device.name
        local_name = advertisement.local_name

        if (
            device_name == "NanoBLE"
            or local_name == "NanoBLE"
        ):
            return device

    return None


async def main():
    print("Loading model weights...")

    model = FallDetectorNN()

    model.load_state_dict(
        torch.load(
            "sisfall_heavy_model.pth",
            map_location="cpu",
        )
    )

    model.eval()

    nano = await find_nano()

    if nano is None:
        print("Could not find the Arduino.")
        return

    print(
        f"Found NanoBLE at {nano.address}. "
        "Connecting..."
    )

    high_prediction_count = 0
    last_alert_time = -float("inf")
    last_inference_sample = 0
    inference_count = 0

    async with BleakClient(nano.address) as client:
        print(
            "Connected! Starting live inference.\n"
        )

        await client.start_notify(
            CHAR_UUID,
            parse_packet,
        )

        try:
            while True:
                await asyncio.sleep(0.1)

                if len(window_buffer) < WINDOW_SIZE:
                    print(
                        f"Collecting samples: "
                        f"{len(window_buffer)}/{WINDOW_SIZE}"
                    )
                    continue

                if (
                    sample_count - last_inference_sample
                    < INFERENCE_EVERY_N_SAMPLES
                ):
                    continue

                last_inference_sample = sample_count
                inference_count += 1

                window = np.asarray(
                    list(window_buffer),
                    dtype=np.float32,
                )

                input_tensor = torch.tensor(
                    window[None, :, :],
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                adl_probability = (
                    probabilities[0, 0].item()
                )

                fall_probability = (
                    probabilities[0, 1].item()
                )

                acceleration = window[:, :3]
                gyroscope = window[:, 3:]

                acceleration_magnitude = np.linalg.norm(
                    acceleration,
                    axis=1,
                )

                gyroscope_magnitude = np.linalg.norm(
                    gyroscope,
                    axis=1,
                )

                if (
                    first_device_timestamp is not None
                    and last_device_timestamp is not None
                    and last_device_timestamp
                    > first_device_timestamp
                ):
                    elapsed_seconds = (
                        last_device_timestamp
                        - first_device_timestamp
                    ) / 1000.0

                    measured_rate = (
                        sample_count / elapsed_seconds
                    )
                else:
                    measured_rate = 0.0

                print(
                    f"Inference {inference_count:04d} | "
                    f"P(ADL)={adl_probability:.3f} | "
                    f"P(FALL)={fall_probability:.3f} | "
                    f"rate={measured_rate:.1f} Hz | "
                    f"a_mag mean="
                    f"{acceleration_magnitude.mean():.3f} | "
                    f"a_mag std="
                    f"{acceleration_magnitude.std():.3f} | "
                    f"gyro mean="
                    f"{gyroscope_magnitude.mean():.3f}"
                )

                if (
                    fall_probability
                    >= FALL_PROBABILITY_THRESHOLD
                ):
                    high_prediction_count += 1
                else:
                    high_prediction_count = 0

                current_time = time.monotonic()

                if (
                    high_prediction_count
                    >= REQUIRED_HIGH_PREDICTIONS
                    and current_time - last_alert_time
                    >= ALERT_COOLDOWN_SECONDS
                ):
                    print(
                        "\n*** FALL DETECTED ***\n"
                    )

                    last_alert_time = current_time
                    high_prediction_count = 0

        except KeyboardInterrupt:
            print("\nStopping pipeline.")

        finally:
            await client.stop_notify(CHAR_UUID)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPipeline stopped.")