# GracefulFall

Adaptive wearable fall detection under sensor and resource degradation.

GracefulFall uses an Arduino Nano 33 BLE Sense Rev2 and a Mac host to detect falls using IMU and audio data. A runtime controller monitors sensor availability and selects the best feasible detection pipeline.

The project evaluates whether adaptive pipeline selection can preserve fall-detection capability when sensors, BLE, or compute resources degrade.

## System architecture

- **Arduino Nano 33 BLE Sense Rev2**
  - BMI270/BMM150 IMU
  - Built-in PDM microphone
  - BLE IMU streaming
  - USB audio streaming
- **Mac host**
  - PyTorch inference
  - Audio/IMU fusion
  - Runtime controller
  - Fault-injection experiments
  - CSV logging

## Pipeline definitions

Pipelines are ordered from highest capability to degraded fallback:

| Pipeline | Sensors | Execution | Purpose | Status |
|---|---|---|---|---|
| **A: `full_mac`** | IMU + audio + PPG | Mac | Full multimodal detection with cardiac context | Planned |
| **B: `imu_audio_mac`** | IMU + audio | Mac | Primary multimodal fall detector | Implemented prototype |
| **C: `imu_mac`** | IMU | Mac | Neural IMU fallback when audio is unavailable | Implemented prototype |
| **D: `local_imu`** | IMU | Nano | On-device model when Mac execution is unavailable | Planned |
| **E: `imu_threshold`** | IMU | Nano or Mac | Simple threshold fallback | Planned |
| **F: `audio_mac`** | Audio | Mac | Audio-only fallback when IMU is unavailable | Implemented prototype |

The controller currently selects among Pipelines B, C, and F.

Typical degradation paths are:

```text
A → B → C → D → E → F