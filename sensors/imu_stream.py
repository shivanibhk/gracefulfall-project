from collections import deque
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class IMUStats:
    sample_count: int = 0
    malformed_packets: int = 0
    first_timestamp: int | None = None
    last_timestamp: int | None = None


class IMUBuffer:
    def __init__(
        self,
        window_size=100,
        sample_logger=None,
    ):
        self.buffer = deque(
            maxlen=window_size
        )

        self.stats = IMUStats()
        self.sample_logger = sample_logger

    def parse_packet(self, sender, data):
        """
        BLE packet format:

        timestamp,ax,ay,az,gx,gy,gz
        """

        try:
            decoded = data.decode(
                "utf-8"
            ).strip()

            parts = decoded.split(",")

            if len(parts) != 7:
                self.stats.malformed_packets += 1
                return

            timestamp = int(parts[0])

            values = [
                float(value)
                for value in parts[1:]
            ]

            if self.stats.first_timestamp is None:
                self.stats.first_timestamp = timestamp

            self.stats.last_timestamp = timestamp
            self.stats.sample_count += 1

            self.buffer.append(values)
            if self.sample_logger is not None:
                self.sample_logger({
                    "host_timestamp": time.time(),
                    "device_timestamp_ms": timestamp,
                    "ax": values[0],
                    "ay": values[1],
                    "az": values[2],
                    "gx": values[3],
                    "gy": values[4],
                    "gz": values[5],
                })

        except (UnicodeDecodeError, ValueError):
            self.stats.malformed_packets += 1

    def ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def snapshot(self):
        if not self.ready():
            return None

        return np.asarray(
            list(self.buffer),
            dtype=np.float32,
        )

    def measured_rate_hz(self):
        first = self.stats.first_timestamp
        last = self.stats.last_timestamp

        if first is None or last is None:
            return 0.0

        elapsed_seconds = (last - first) / 1000.0

        if elapsed_seconds <= 0:
            return 0.0

        return self.stats.sample_count / elapsed_seconds