import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from parse_sisfall import load_sisfall_data

class SisFallDataset(Dataset):
    def __init__(self, data_windows, labels):
        # data_windows shape: (Num_Windows, 100_samples, 6_channels)
        # labels shape: (Num_Windows,) -> 0 for ADL, 1 for Fall
        self.X = torch.tensor(data_windows, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Update in_channels to 6 (Accel + Gyro)
class FallDetectorNN(nn.Module):
    def __init__(self):
        super(FallDetectorNN, self).__init__()
        # SisFall has 6 channels: Accel X, Y, Z + Gyro X, Y, Z
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 25, 64), # Window of 100 -> pooled to 50 -> pooled to 25
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents overfitting on the specific SisFall subjects
            nn.Linear(64, 2) # [ADL, Fall]
        )

    def forward(self, x):
        # PyTorch expects (Batch, Channels, Sequence_Length)
        x = x.transpose(1, 2) 
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# loop that adjusts the model weights
def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_loss = 0
        correct = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() # Clear old gradients
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass (calculate gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            
        accuracy = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
    
    # Save the trained weights to a file so Pipeline A can load them later!
    torch.save(model.state_dict(), "sisfall_heavy_model.pth")
    print("Model saved to sisfall_heavy_model.pth")

# 1. Run the parser on your downloaded folder
print("Parsing SisFall raw files. This might take a minute...")
# Update this path to match your folder hierarchy
train_X, train_y = load_sisfall_data("./SisFall/SisFall_dataset")

# 2. Load the parsed arrays into PyTorch
dataset = SisFallDataset(train_X, train_y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Initialize the 6-channel Neural Network and train!
my_model = FallDetectorNN()
train_model(my_model, dataloader, num_epochs=15)