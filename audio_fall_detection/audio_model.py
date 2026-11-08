import librosa
import numpy as np
import torch
import torch.nn as nn


TARGET_SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SAMPLE_RATE * CLIP_SECONDS)

#convert wav file to log-mel spectogram
def wav_to_log_mel(path):
    audio, _ = librosa.load(
        path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    #pad audio with silence at end if not long enough
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(
            audio,
            (0, CLIP_SAMPLES - len(audio)),
        )
    else: #if longer, only use first 3 seconds
        audio = audio[:CLIP_SAMPLES]

    audio = audio.astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
        fmin=50,
        fmax=8000,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    log_mel = (
        log_mel - log_mel.mean()
    ) / (log_mel.std() + 1e-6)

    return log_mel.astype(np.float32)

def audio_rms(path):
    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    return float(
        np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    )

class AudioFallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)