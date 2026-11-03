import asyncio
import time
import torch
import torch.nn as nn
import numpy as np
from bleak import BleakClient, BleakScanner
from collections import deque

# --- 1. Recreate the Model Architecture ---
class FallDetectorNN(nn.Module):
    def __init__(self):
        super(FallDetectorNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 25, 64), 
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# --- 2. Load the Trained Model ---
print("Loading model weights...")
model = FallDetectorNN()
model.load_state_dict(torch.load("sisfall_heavy_model.pth"))
model.eval() # Set to evaluation mode

# --- 3. BLE Setup & Live Buffer ---
CHAR_UUID = "00002A56-0000-1000-8000-00805f9b34fb"
# Deque automatically drops the oldest item when it reaches maxlen
window_buffer = deque(maxlen=100) 

def handle_data(sender, data):
    decoded = data.decode('utf-8').strip()
    parts = decoded.split(',')
    
    # We expect 7 items: timestamp, ax, ay, az, gx, gy, gz
    if len(parts) == 7:
        try:
            # Extract the 6 sensor channels (skip timestamp)
            features = [float(val) for val in parts[1:]]
            window_buffer.append(features)
            
            # Only run inference if we have a full 2-second window
            if len(window_buffer) == 100:
                # Convert buffer to tensor: shape (Batch=1, Samples=100, Channels=6)
                input_tensor = torch.tensor([list(window_buffer)], dtype=torch.float32)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    
                if prediction == 1:
                    print(f"[{parts[0]}] 🚨 FALL DETECTED! 🚨")
                else:
                    print(f"[{parts[0]}] ADL (Normal)")
                    
        except ValueError:
            pass # Ignore corrupted strings during BLE transmission

async def main():
    print("Scanning for NanoBLE...")
    devices = await BleakScanner.discover()
    nano = next((d for d in devices if d.name == "NanoBLE"), None)
    
    if not nano:
        print("Could not find the Arduino.")
        return
        
    print(f"Found {nano.name}. Connecting...")
    async with BleakClient(nano.address) as client:
        print("Connected! Starting Pipeline C live inference...\n")
        await client.start_notify(CHAR_UUID, handle_data)
        
        # Run continuously until you press Ctrl+C
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())