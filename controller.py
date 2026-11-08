from dataclasses import dataclass


@dataclass
class SensorHealth:
    imu_available: bool
    audio_available: bool
    ppg_available: bool

    imu_rate_hz: float
    audio_rate_hz: float

    ble_latency_ms: float
    cpu_load: float

    mac_available: bool = True
    nano_model_available: bool = False


class PipelineController:
    """
    Selects the best feasible pipeline.

    Current active pipelines:

        B = IMU + audio on Mac
        C = IMU-only neural model on Mac
        F = audio-only model on Mac

    Future pipelines:

        A = IMU + audio + PPG on Mac
        D = local IMU model on Nano
        E = IMU threshold fallback
    """

    MIN_IMU_RATE_HZ = 30.0
    MIN_AUDIO_RATE_HZ = 8000.0

    def __init__(self):
        self.current_pipeline = None
        self.last_switch_reason = None

    def choose(self, health):
        # Future full pipeline.
        if (
            health.imu_available
            and health.audio_available
            and health.ppg_available
            and health.mac_available
            and health.imu_rate_hz >= 45.0
            and health.audio_rate_hz >= 14000.0
            and health.cpu_load < 0.80
        ):
            return "A"

        # Current best pipeline.
        if (
            health.imu_available
            and health.audio_available
            and health.mac_available
            and health.imu_rate_hz >= self.MIN_IMU_RATE_HZ
            and health.audio_rate_hz >= self.MIN_AUDIO_RATE_HZ
            and health.cpu_load < 0.90
        ):
            return "B"

        # IMU-only neural model on the Mac.
        if (
            health.imu_available
            and health.mac_available
            and health.imu_rate_hz >= self.MIN_IMU_RATE_HZ
        ):
            return "C"

        # Lightweight Nano model.
        if (
            health.imu_available
            and health.nano_model_available
        ):
            return "D"

        # Simple threshold fallback.
        if health.imu_available:
            return "E"

        # Emergency Audio-only fallback.
        if (
            health.audio_available
            and health.mac_available
            and health.audio_rate_hz >= self.MIN_AUDIO_RATE_HZ
        ):
            return "F"

        return "UNAVAILABLE"

    def update(self, health):
        selected = self.choose(health)

        switched = selected != self.current_pipeline

        if switched:
            previous = self.current_pipeline
            self.current_pipeline = selected

            self.last_switch_reason = (
                f"{previous} -> {selected}"
            )

        return {
            "pipeline": self.current_pipeline,
            "switched": switched,
            "reason": self.last_switch_reason,
        }