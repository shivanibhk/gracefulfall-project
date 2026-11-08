import torch
import torch.nn as nn


class FallDetectorNN(nn.Module):
    """
    SisFall IMU fall-detection model.

    Input shape:
        batch x 100 samples x 6 channels

    Channels:
        ax, ay, az, gx, gy, gz

    Output classes:
        class 0 = normal activity
        class 1 = fall
    """

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 25, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # Input shape:
        # batch x samples x channels

        # Conv1d requires:
        # batch x channels x samples
        x = x.transpose(1, 2)

        x = self.conv_layers(x)

        x = x.reshape(x.size(0), -1)

        return self.fc_layers(x)

def load_imu_model(model_path):
    device = torch.device("cpu")

    model = FallDetectorNN()

    checkpoint = torch.load(
        Path(model_path),
        map_location=device,
    )

    # Supports either a raw state dictionary or
    # a checkpoint containing "model_state".
    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model