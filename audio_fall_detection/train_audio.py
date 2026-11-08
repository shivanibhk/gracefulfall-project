import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from audio_model import AudioFallCNN, wav_to_log_mel


class SafeDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, _ = self.records[index]

        mel = wav_to_log_mel(path)

        features = torch.tensor(
            mel[None, :, :],
            dtype=torch.float32,
        )

        target = torch.tensor(
            label,
            dtype=torch.long,
        )

        return features, target


def collect_safe_files(root):
    records = []

    for path in sorted(Path(root).glob("*.wav")):
        parts = path.stem.split("-")

        if len(parts) != 5:
            print(f"Skipping unexpected filename: {path.name}")
            continue

        fold = int(parts[0])
        class_code = parts[-1]

        if class_code == "01":
            label = 1
        elif class_code == "02":
            label = 0
        else:
            print(f"Skipping unknown label: {path.name}")
            continue

        records.append((str(path), label, fold))

    return records


def split_by_fold(records):
    train_records = [
        record for record in records
        if record[2] <= 8
    ]

    validation_records = [
        record for record in records
        if record[2] == 9
    ]

    test_records = [
        record for record in records
        if record[2] == 10
    ]

    return train_records, validation_records, test_records


def evaluate(model, loader, device):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            batch_predictions = logits.argmax(dim=1)

            predictions.extend(
                batch_predictions.cpu().numpy()
            )

            labels.extend(
                targets.cpu().numpy()
            )

    print(
        classification_report(
            labels,
            predictions,
            target_names=["normal", "fall"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))

    return np.mean(
        np.asarray(predictions) == np.asarray(labels)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="audio-data/SAFE_raw",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--output",
        default="audio-fall-detection/audio_model.pth",
    )

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    records = collect_safe_files(args.data)

    train_records, validation_records, test_records = (
        split_by_fold(records)
    )

    print(f"Total files: {len(records)}")
    print(f"Training files: {len(train_records)}")
    print(f"Validation files: {len(validation_records)}")
    print(f"Test files: {len(test_records)}")

    train_dataset = SafeDataset(train_records)
    validation_dataset = SafeDataset(validation_records)
    test_dataset = SafeDataset(test_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = AudioFallCNN().to(device)

    class_counts = np.bincount(
        [record[1] for record in train_records],
        minlength=2,
    )

    weights = len(train_records) / (
        3.0 * np.maximum(class_counts, 1)
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Training loss: {running_loss:.3f}")

        print("Validation performance:")
        evaluate(model, validation_loader, device)

    print("\nFinal test performance:")
    evaluate(model, test_loader, device)

    checkpoint = {
        "model_state": model.state_dict(),
        "sample_rate": 16000,
        "clip_seconds": 3.0,
    }

    torch.save(checkpoint, args.output)

    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()