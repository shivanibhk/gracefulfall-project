import argparse
import csv
from pathlib import Path

import torch

from audio_model import AudioFallCNN, wav_to_log_mel


def get_expected_label(path):
    """
    Supports both:

    SAFE: nano-like-name-01.wav
    Nano: nano_1_01.wav

    01 = fall
    02 = normal
    """

    code = path.stem.replace("-", "_").split("_")[-1]

    if code == "01":
        return 1

    if code == "02":
        return 0

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--folder",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="batch_results.csv",
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    checkpoint = torch.load(
        args.model,
        map_location=device,
    )

    model = AudioFallCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    audio_files = sorted(
        Path(args.folder).glob("*.wav")
    )

    if not audio_files:
        raise RuntimeError(
            f"No WAV files found in {args.folder}"
        )

    total = 0
    correct = 0

    with open(
        args.output,
        "w",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "file",
            "expected_label",
            "expected_class",
            "normal_probability",
            "fall_probability",
            "predicted_label",
            "predicted_class",
            "correct",
        ])

        for audio_path in audio_files:
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

            normal_probability = (
                probabilities[0, 0].item()
            )

            fall_probability = (
                probabilities[0, 1].item()
            )

            predicted_label = int(
                fall_probability >= 0.80
            )

            expected_label = get_expected_label(
                audio_path
            )

            predicted_class = (
                "fall"
                if predicted_label == 1
                else "normal"
            )

            expected_class = (
                "unknown"
                if expected_label is None
                else (
                    "fall"
                    if expected_label == 1
                    else "normal"
                )
            )

            is_correct = (
                expected_label is not None
                and predicted_label == expected_label
            )

            if expected_label is not None:
                total += 1
                correct += int(is_correct)

            print(
                f"{audio_path.name}: "
                f"expected={expected_class}, "
                f"predicted={predicted_class}, "
                f"P(fall)={fall_probability:.3f}, "
                f"correct={is_correct}"
            )

            writer.writerow([
                audio_path.name,
                expected_label,
                expected_class,
                f"{normal_probability:.6f}",
                f"{fall_probability:.6f}",
                predicted_label,
                predicted_class,
                is_correct,
            ])

    if total > 0:
        accuracy = correct / total

        print()
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.3f}")

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()