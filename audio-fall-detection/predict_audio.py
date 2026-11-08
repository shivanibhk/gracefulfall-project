import argparse

import librosa
import torch

from audio_model import (
    AudioFallCNN,
    wav_to_log_mel,
)


def load_model(model_path):
    device = torch.device("cpu")

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model = AudioFallCNN()
    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    return model


def audio_rms(audio_path):
    audio, _ = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    return float(
        (
            (audio.astype("float64") ** 2)
            .mean()
        ) ** 0.5
    )


def predict_file(
    model,
    audio_path,
    rms_threshold=0.002,
    fall_threshold=0.80,
):
    rms = audio_rms(audio_path)

    # Treat extremely quiet audio as normal.
    if rms < rms_threshold:
        return {
            "normal_probability": 1.0,
            "fall_probability": 0.0,
            "predicted_label": 0,
            "predicted_class": "normal",
            "rms": rms,
            "quiet_audio": True,
        }

    mel = wav_to_log_mel(audio_path)

    features = torch.tensor(
        mel[None, None, :, :],
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(features)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    normal_probability = (
        probabilities[0, 0].item()
    )

    fall_probability = (
        probabilities[0, 1].item()
    )

    predicted_label = int(
        fall_probability >= fall_threshold
    )

    predicted_class = (
        "fall"
        if predicted_label == 1
        else "normal"
    )

    return {
        "normal_probability": normal_probability,
        "fall_probability": fall_probability,
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "rms": rms,
        "quiet_audio": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--audio",
        required=True,
    )

    args = parser.parse_args()

    model = load_model(args.model)

    result = predict_file(
        model,
        args.audio,
    )

    print(f"Audio file: {args.audio}")
    print(f"Audio RMS: {result['rms']:.6f}")
    print(
        "P(normal) = "
        f"{result['normal_probability']:.3f}"
    )
    print(
        "P(fall)   = "
        f"{result['fall_probability']:.3f}"
    )
    print(
        "Prediction: "
        f"{result['predicted_class'].upper()}"
    )


if __name__ == "__main__":
    main()