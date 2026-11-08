import argparse
import struct
import time
import wave

import numpy as np
import serial


MAGIC = 0xA55A
SAMPLE_RATE = 16000

# Header:
# uint16 magic
# uint32 timestamp
# uint16 sample count
HEADER_FORMAT = "<HIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def read_exactly(connection, count):
    data = bytearray()

    while len(data) < count:
        block = connection.read(count - len(data))

        if not block:
            raise RuntimeError(
                "Serial connection closed before enough data arrived"
            )

        data.extend(block)

    return bytes(data)


def find_frame(connection):
    magic_bytes = struct.pack("<H", MAGIC)

    while True:
        first = connection.read(1)

        if not first:
            raise RuntimeError("Serial connection closed")

        if first != magic_bytes[:1]:
            continue

        second = connection.read(1)

        if second == magic_bytes[1:2]:
            return


def record_audio(port, output_path, seconds):
    connection = serial.Serial(
        port=port,
        baudrate=921600,
        timeout=3,
    )

    print("Waiting briefly for the Nano to begin streaming...")
    time.sleep(2.0)

    print(
        f"Recording {seconds:.1f} seconds "
        f"to {output_path}"
    )

    samples = []
    deadline = time.time() + seconds

    while time.time() < deadline:
        find_frame(connection)

        # We already consumed the 2-byte magic number.
        # Read the remaining timestamp and sample-count bytes.
        header_rest = read_exactly(
            connection,
            HEADER_SIZE - 2,
        )

        timestamp, sample_count = struct.unpack(
            "<IH",
            header_rest,
        )

        # int16_t uses exactly 2 bytes per sample.
        payload = read_exactly(
            connection,
            sample_count * 2,
        )

        chunk = np.frombuffer(
            payload,
            dtype="<i2",
        ).copy()

        samples.append(chunk)

    connection.close()

    if not samples:
        raise RuntimeError("No audio frames received")

    audio = np.concatenate(samples)

    target_count = int(seconds * SAMPLE_RATE)
    audio = audio[:target_count]

    with wave.open(output_path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(
            audio.astype("<i2").tobytes()
        )

    print(
        f"Saved {len(audio) / SAMPLE_RATE:.2f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    record_audio(
        port=args.port,
        output_path=args.output,
        seconds=args.seconds,
    )