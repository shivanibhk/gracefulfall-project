def fuse_predictions(
    imu_result,
    audio_result,
):
    """
    Combine IMU and audio probabilities.

    Both probabilities use:

        0.0 = normal
        1.0 = fall
    """

    available_scores = []
    available_weights = []

    if (
        imu_result is not None
        and imu_result.get("usable", False)
    ):
        available_scores.append(
            imu_result["fall_probability"]
        )

        # IMU is currently the more trustworthy sensor
        #70-30 split is random right now
        available_weights.append(0.70)

    if (
        audio_result is not None
        and audio_result.get("usable", False)
    ):
        available_scores.append(
            audio_result["fall_probability"]
        )

        available_weights.append(0.30)

    if not available_scores:
        return {
            "usable": False,
            "fall_probability": 0.0,
            "decision": "unavailable",
        }

    total_weight = sum(
        available_weights
    )

    combined_probability = sum(
        score * weight
        for score, weight in zip(
            available_scores,
            available_weights,
        )
    ) / total_weight

    return {
        "usable": True,
        "fall_probability": combined_probability,
        "decision": (
            "fall"
            if combined_probability >= 0.80
            else "normal"
        ),
    }