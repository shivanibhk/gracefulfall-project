import struct
import threading
import time
from collections import deque

import numpy as np
import serial

MAGIC = 0xA55A
HEADER_FORMAT = "<HIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_FRAME_SAMPLES = 512


class AudioStream:
    """
    Reads binary microphone frames from the Nano
    over USB Serial.

    The stream does not begin until start_streaming()
    sends START to the Nano.
    """

    def __init__(
        self,
        port,
        sample_rate=16000,
        window_seconds=3.0,
        frame_filter=None,
        frame_delay=None,
    ):
        self.port = port
        self.sample_rate = sample_rate
        self.window_size = int(
            sample_rate * window_seconds
        )

        self.frame_filter = frame_filter
        self.frame_delay = frame_delay

        self.buffer = deque(
            maxlen=self.window_size
        )

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.connection = None
        self.thread = None

        self.frames_received = 0
        self.samples_received = 0
        self.dropped_frames = 0
        self.invalid_frames = 0
        self.last_device_timestamp = None
        self.error = None

        self.frame_number = 0

    def open(self):
        self.connection = serial.Serial(
            port=self.port,
            baudrate=921600,
            timeout=0.1,
        )

        # Opening USB serial may reset the Nano.
        time.sleep(1.0)

        # Remove startup text and incomplete data.
        self.connection.reset_input_buffer()

    def start_streaming(self):
        if self.connection is None:
            raise RuntimeError(
                "Call open() before start_streaming()"
            )

        self.connection.write(
            b"START\n"
        )

        deadline = (
            time.monotonic() + 2.0
        )

        while time.monotonic() < deadline:
            line = self.connection.readline()

            if b"STREAM_STARTED" in line:
                break

        else:
            raise RuntimeError(
                "Nano did not acknowledge START"
            )

        self.reset_session()

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
        )

        self.thread.start()

    def reset_session(self):
        with self.lock:
            self.buffer.clear()

            self.frames_received = 0
            self.samples_received = 0
            self.invalid_frames = 0
            self.last_device_timestamp = None
            self.error = None

    def stop(self):
        self.stop_event.set()

        if self.connection is not None:
            try:
                self.connection.write(
                    b"STOP\n"
                )
            except Exception:
                pass

            self.connection.close()

        if self.thread is not None:
            self.thread.join(
                timeout=2.0
            )

    def ready(self):
        with self.lock:
            return (
                len(self.buffer)
                >= self.window_size
            )

    def snapshot(self):
        with self.lock:
            if len(self.buffer) < self.window_size:
                return None

            return np.asarray(
                list(self.buffer),
                dtype=np.int16,
            )

    def stats(self):
        with self.lock:
            return {
                "frames_received": self.frames_received,
                "samples_received": self.samples_received,
                "dropped_frames": self.dropped_frames,
                "invalid_frames": self.invalid_frames,
                "last_device_timestamp": (
                    self.last_device_timestamp
                ),
                "buffer_samples": len(self.buffer),
                "error": self.error,
            }

    def _read_loop(self):
        try:
            while not self.stop_event.is_set():
                frame = self._read_frame()

                if frame is None:
                    continue

                timestamp, samples = frame

                self.frame_number += 1

                if self.frame_delay is not None:
                    self.frame_delay()

                if (
                    self.frame_filter is not None
                    and not self.frame_filter(
                        self.frame_number
                    )
                ):
                    with self.lock:
                        self.dropped_frames += 1

                    continue

                with self.lock:
                    self.buffer.extend(
                        samples.tolist()
                    )

                    self.frames_received += 1
                    self.samples_received += len(samples)
                    self.last_device_timestamp = timestamp

        except Exception as error:
            with self.lock:
                self.error = repr(error)

    def _read_frame(self):
        magic_bytes = struct.pack(
            "<H",
            MAGIC,
        )

        while not self.stop_event.is_set():
            first = self.connection.read(1)

            if not first:
                continue

            if first != magic_bytes[:1]:
                continue

            second = self.connection.read(1)

            if second != magic_bytes[1:2]:
                continue

            header_rest = self._read_exactly(
                HEADER_SIZE - 2
            )

            if header_rest is None:
                return None

            timestamp, sample_count = struct.unpack(
                "<IH",
                header_rest,
            )

            if not (
                0
                < sample_count
                <= MAX_FRAME_SAMPLES
            ):
                with self.lock:
                    self.invalid_frames += 1

                return None

            payload = self._read_exactly(
                sample_count * 2
            )

            if payload is None:
                return None

            samples = np.frombuffer(
                payload,
                dtype="<i2",
            ).copy()

            return timestamp, samples

        return None

    def _read_exactly(self, count):
        data = bytearray()

        while (
            len(data) < count
            and not self.stop_event.is_set()
        ):
            block = self.connection.read(
                count - len(data)
            )

            if block:
                data.extend(block)

        if len(data) != count:
            return None

        return bytes(data)