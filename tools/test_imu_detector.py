import numpy as np

from detectors.imu_detector import predict_imu
from imu.imu_model import FallDetectorNN


model = FallDetectorNN()

window = np.random.randn(
    100,
    6,
).astype(np.float32)

result = predict_imu(
    model=model,
    window=window,
)

print("IMU detector smoke test passed")
print(result)