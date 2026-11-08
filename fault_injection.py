import time
from dataclasses import dataclass


@dataclass
class FaultProfile:
    audio_enabled: bool = True
    imu_enabled: bool = True

    audio_drop_every: int = 0
    imu_drop_every: int = 0

    audio_delay_ms: float = 0.0
    imu_delay_ms: float = 0.0

    inference_delay_ms: float = 0.0

    def keep_audio_frame(self, frame_number):
        if not self.audio_enabled:
            return False

        return not (
            self.audio_drop_every > 0
            and frame_number % self.audio_drop_every == 0
        )

    def keep_imu_packet(self, packet_number):
        if not self.imu_enabled:
            return False

        return not (
            self.imu_drop_every > 0
            and packet_number % self.imu_drop_every == 0
        )

    def delay_audio(self):
        if self.audio_delay_ms > 0:
            time.sleep(
                self.audio_delay_ms / 1000.0
            )

    def delay_imu(self):
        if self.imu_delay_ms > 0:
            time.sleep(
                self.imu_delay_ms / 1000.0
            )

    def delay_inference(self):
        if self.inference_delay_ms > 0:
            time.sleep(
                self.inference_delay_ms / 1000.0
            )


class FaultAwareIMUCallback:
    """
    Adapter between Bleak and IMUBuffer.

    Bleak calls this object whenever an IMU
    notification arrives.
    """

    def __init__(
        self,
        imu_buffer,
        fault_profile,
    ):
        self.imu_buffer = imu_buffer
        self.fault_profile = fault_profile

        self.packet_number = 0
        self.accepted_packets = 0
        self.dropped_packets = 0

    def __call__(self, sender, data):
        self.packet_number += 1

        self.fault_profile.delay_imu()

        keep_packet = (
            self.fault_profile.keep_imu_packet(
                self.packet_number
            )
        )

        if not keep_packet:
            self.dropped_packets += 1
            return

        self.accepted_packets += 1

        self.imu_buffer.parse_packet(
            sender,
            data,
        )

    def stats(self):
        return {
            "packets_seen": self.packet_number,
            "packets_accepted": (
                self.accepted_packets
            ),
            "packets_dropped": (
                self.dropped_packets
            ),
        }