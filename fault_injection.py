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