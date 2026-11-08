from pathlib import Path

import torch

from audio_fall_detection.audio_model import (
    AudioFallCNN,
    audio_to_log_mel,
    wav_to_log_mel,
)


def load_audio_model(model_path):
    model = AudioFallCNN()

    checkpoint = torch.load(
        Path(model_path),
        map_location="cpu",
    )

    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def predict_audio(model, audio_path):
    """
    Predict one WAV file.

    The WAV preprocessing must be identical
    to the preprocessing used during training.
    """

    mel = wav_to_log_mel(
        str(audio_path)
    )

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

    return {
        "normal_probability": float(
            probabilities[0, 0].item()
        ),
        "fall_probability": float(
            probabilities[0, 1].item()
        ),
        "usable": True,
        "audio_path": str(audio_path),
    }

def predict_audio_samples(model, samples):
    """
    Predict one rolling raw-audio window.

    The Nano provides signed int16 PCM.
    WAV loading normally provides floats in [-1, 1],
    so the serial samples must be scaled.
    """

    audio = (
        samples.astype("<f4")
        / 32768.0
    )

    mel = audio_to_log_mel(audio)

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

    return {
        "normal_probability": float(
            probabilities[0, 0].item()
        ),
        "fall_probability": float(
            probabilities[0, 1].item()
        ),
        "usable": True,
    }