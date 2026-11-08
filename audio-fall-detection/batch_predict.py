import argparse
import csv
from pathlib import Path

from predict_audio import (
    load_model,
    predict_file,
)


def get_expected_label(path):
    """
    Supports:

    SAFE: 01-020-02-073-01.wav
    Nano: nano_1_01.wav

    01 = fall
    02 = normal
    """

    code = (
        path.stem
        .replace("-", "_")
        .split("_")[-1]
    )

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

    model = load_model(args.model)

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
            "rms",
            "expected_label",
            "expected_class",
            "normal_probability",
            "fall_probability",
            "predicted_label",
            "predicted_class",
            "quiet_audio",
            "correct",
        ])

        for audio_path in audio_files:
            result = predict_file(
                model,
                str(audio_path),
            )

            expected_label = (
                get_expected_label(audio_path)
            )

            if expected_label is None:
                expected_class = "unknown"
            elif expected_label == 1:
                expected_class = "fall"
            else:
                expected_class = "normal"

            is_correct = (
                expected_label is not None
                and result["predicted_label"]
                == expected_label
            )

            if expected_label is not None:
                total += 1
                correct += int(is_correct)

            print(
                f"{audio_path.name}: "
                f"expected={expected_class}, "
                f"predicted="
                f"{result['predicted_class']}, "
                f"P(fall)="
                f"{result['fall_probability']:.3f}, "
                f"RMS={result['rms']:.6f}, "
                f"quiet="
                f"{result['quiet_audio']}, "
                f"correct={is_correct}"
            )

            writer.writerow([
                audio_path.name,
                f"{result['rms']:.6f}",
                expected_label,
                expected_class,
                f"{result['normal_probability']:.6f}",
                f"{result['fall_probability']:.6f}",
                result["predicted_label"],
                result["predicted_class"],
                result["quiet_audio"],
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