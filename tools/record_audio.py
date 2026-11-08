import argparse
import struct
import time
import wave

import numpy as np
import serial


MAGIC = 0xA55A
SAMPLE_RATE = 16000
MAX_FRAME_SAMPLES = 512

HEADER_FORMAT = "<HIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def read_exactly(connection, count):
    data = bytearray()

    while len(data) < count:
        block = connection.read(
            count - len(data)
        )

        if not block:
            raise RuntimeError(
                "Serial connection closed"
            )

        data.extend(block)

    return bytes(data)


def find_frame(connection):
    """
    Search the serial stream for the two-byte
    frame magic number.

    The firmware sends:

        0x5A 0xA5

    because 0xA55A is transmitted little-endian.
    """

    magic_bytes = struct.pack(
        "<H",
        MAGIC,
    )

    while True:
        first = connection.read(1)

        if not first:
            raise RuntimeError(
                "Serial connection closed"
            )

        if first != magic_bytes[:1]:
            continue

        second = connection.read(1)

        if second == magic_bytes[1:2]:
            return


def read_audio_frame(connection):
    """
    Read one complete binary audio frame.

    Firmware frame:

        2 bytes: magic
        4 bytes: timestamp
        2 bytes: sample count
        N bytes: signed 16-bit PCM samples
    """

    find_frame(connection)

    # The magic number was already consumed.
    # Read timestamp + sample count.
    header_rest = read_exactly(
        connection,
        HEADER_SIZE - 2,
    )

    timestamp, sample_count = struct.unpack(
        "<IH",
        header_rest,
    )

    if (
        sample_count <= 0
        or sample_count > MAX_FRAME_SAMPLES
    ):
        raise RuntimeError(
            "Invalid audio frame sample count: "
            f"{sample_count}"
        )

    payload = read_exactly(
        connection,
        sample_count * 2,
    )

    samples = np.frombuffer(
        payload,
        dtype="<i2",
    ).copy()

    return timestamp, samples


def record_audio(port, output_path, seconds):
    target_count = int(
        seconds * SAMPLE_RATE
    )

    connection = serial.Serial(
        port=port,
        baudrate=921600,
        timeout=3,
    )

    try:
        print(
            "Waiting briefly for the Nano to reset..."
        )

        # Opening the serial port may reset the Nano.
        time.sleep(1.0)

        # Discard startup text and incomplete old frames.
        # We do not need AUDIO_READY because the binary
        # parser searches for the magic number.
        connection.reset_input_buffer()

        print(
            f"Recording {seconds:.2f} seconds "
            f"({target_count} samples) to "
            f"{output_path}"
        )

        samples = []
        collected_count = 0

        # Allow extra time for serial startup or
        # occasional delayed frames.
        deadline = (
            time.time()
            + seconds
            + 3.0
        )

        while (
            collected_count < target_count
            and time.time() < deadline
        ):
            _, chunk = read_audio_frame(
                connection
            )

            samples.append(chunk)
            collected_count += len(chunk)

            print(
                f"\rCollected "
                f"{min(collected_count, target_count)}"
                f"/{target_count} samples",
                end="",
                flush=True,
            )

        print()

    finally:
        connection.close()

    if not samples:
        raise RuntimeError(
            "No audio frames received"
        )

    audio = np.concatenate(samples)

    if len(audio) < target_count:
        raise RuntimeError(
            "The Nano did not provide enough audio: "
            f"{len(audio)}/{target_count} samples"
        )

    audio = audio[:target_count]

    with wave.open(output_path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(
            audio.astype("<i2").tobytes()
        )

    actual_seconds = (
        len(audio) / SAMPLE_RATE
    )

    print(
        f"Saved {len(audio)} samples "
        f"({actual_seconds:.2f} seconds)"
    )


def main():
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


if __name__ == "__main__":
    main()