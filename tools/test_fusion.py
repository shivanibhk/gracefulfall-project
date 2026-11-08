from detectors.fusion_detector import fuse_predictions


imu_result = {
    "fall_probability": 0.90,
    "usable": True,
}

audio_result = {
    "fall_probability": 0.70,
    "usable": True,
}

result = fuse_predictions(
    imu_result=imu_result,
    audio_result=audio_result,
)

print(result)

assert result["usable"] is True
assert 0.0 <= result["fall_probability"] <= 1.0

print("Fusion test passed")