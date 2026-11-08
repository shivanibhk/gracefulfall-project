import numpy as np
import torch


IMPACT_THRESHOLD_G = 1.8
FREE_FALL_THRESHOLD_G = 0.65
QUIET_STD_THRESHOLD_G = 0.15
ORIENTATION_CHANGE_DEGREES = 45.0


def fall_motion_signature(window):
    """
    Analyze a 100-sample IMU window.

    Columns:
        0: ax
        1: ay
        2: az
        3: gx
        4: gy
        5: gz
    """

    if window.shape != (100, 6):
        raise ValueError(
            f"Expected IMU shape (100, 6), got {window.shape}"
        )

    acceleration = window[:, :3]

    acceleration_magnitude = np.linalg.norm(
        acceleration,
        axis=1,
    )

    free_fall_detected = (
        acceleration_magnitude.min()
        <= FREE_FALL_THRESHOLD_G
    )

    impact_detected = (
        acceleration_magnitude.max()
        >= IMPACT_THRESHOLD_G
    )

    post_event_values = acceleration_magnitude[-20:]

    post_event_quiet = (
        post_event_values.std()
        <= QUIET_STD_THRESHOLD_G
    )

    before = acceleration[:20].mean(axis=0)
    after = acceleration[-20:].mean(axis=0)

    before_norm = np.linalg.norm(before)
    after_norm = np.linalg.norm(after)

    if before_norm > 0 and after_norm > 0:
        cosine = np.dot(before, after) / (
            before_norm * after_norm
        )

        cosine = np.clip(cosine, -1.0, 1.0)

        orientation_change = np.degrees(
            np.arccos(cosine)
        )
    else:
        orientation_change = 0.0

    orientation_changed = (
        orientation_change
        >= ORIENTATION_CHANGE_DEGREES
    )

    strict_fall = (
        free_fall_detected
        and impact_detected
    )

    soft_fall = (
        impact_detected
        and orientation_changed
        and post_event_quiet
    )

    return {
        "fall_signature": (
            strict_fall or soft_fall
        ),
        "free_fall": free_fall_detected,
        "impact": impact_detected,
        "orientation_change": float(
            orientation_change
        ),
        "post_event_quiet": post_event_quiet,
        "accel_min": float(
            acceleration_magnitude.min()
        ),
        "accel_max": float(
            acceleration_magnitude.max()
        ),
    }


def predict_imu(model, window):
    """
    Run the IMU neural model and physical fall checks.
    """

    window = np.asarray(
        window,
        dtype=np.float32,
    )

    if window.shape != (100, 6):
        raise ValueError(
            f"Expected shape (100, 6), got {window.shape}"
        )

    input_tensor = torch.tensor(
        window[None, :, :],
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    motion = fall_motion_signature(window)

    return {
        "normal_probability": float(
            probabilities[0, 0].item()
        ),
        "fall_probability": float(
            probabilities[0, 1].item()
        ),
        "motion": motion,
        "usable": True,
    }