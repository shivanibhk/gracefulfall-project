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
def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train() # Set model to training mode
        total_loss = 0
        train_correct = 0
        
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
            train_correct += (predictions == batch_y).sum().item()
            
        train_accuracy = train_correct / len(train_loader.dataset)
            
        # --- TESTING PHASE (No learning allowed) ---
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for test_X, test_y in test_loader:
                test_outputs = model(test_X)
                test_predictions = torch.argmax(test_outputs, dim=1)
                test_correct += (test_predictions == test_y).sum().item()
                
        test_accuracy = test_correct / len(test_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {total_loss:.2f} | Train Acc: {train_accuracy*100:.2f}% | Test Acc: {test_accuracy*100:.2f}%")
    
    torch.save(model.state_dict(), "sisfall_heavy_model.pth")
    print("Model saved to sisfall_heavy_model.pth")

def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    """A numpy-based replacement for sklearn's train_test_split."""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 1. Run the parser on your downloaded folder
# 1. Run the parser
print("Parsing SisFall raw files. This might take a minute...")
X_all, y_all = load_sisfall_data("./SisFall/SisFall_dataset")

# 2. Split the data using our custom numpy function
X_train, X_test, y_train, y_test = manual_train_test_split(X_all, y_all, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} windows, Testing on {len(X_test)} windows.")

# 3. Create two separate PyTorch Loaders
train_dataset = SisFallDataset(X_train, y_train)
test_dataset = SisFallDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Initialize and train!
my_model = FallDetectorNN()
train_model(my_model, train_loader, test_loader, num_epochs=15)