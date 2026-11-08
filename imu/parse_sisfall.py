import os
import numpy as np

def load_sisfall_data(dataset_folder_path):
    """
    Reads SisFall .txt files, converts bits to physical units, 
    downsamples to 50Hz, and chops into 2-second windows.
    """
    # Conversion constants from the SisFall Readme
    ACCEL_CONVERSION = (2 * 16) / (2**13)   # ADXL345
    GYRO_CONVERSION = (2 * 2000) / (2**16)  # ITG3200
    
    WINDOW_SIZE = 100 # 2 seconds at 50Hz
    STEP_SIZE = 50    # 1 second overlap between windows
    
    all_windows = []
    all_labels = []
    
    # Recursively iterate through all subfolders to find .txt files
    for root, dirs, files in os.walk(dataset_folder_path):
        for filename in files:
            # Skip non-txt files and the Readme file
            if not filename.endswith('.txt') or filename.lower().startswith('readme'):
                continue
                
            filepath = os.path.join(root, filename)
            
            # 1. Determine Label: D (ADL) = 0, F (Fall) = 1
            is_fall = 1 if filename.startswith('F') else 0
            
            # 2. Read the raw text file
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # --- EVERYTHING BELOW MUST BE INDENTED INSIDE THE FILENAME LOOP ---
            parsed_data = []
            for line in lines:
                # SisFall lines look like: " 123, 456, 789, ... ,\n"
                parts = line.strip().strip(',').split(',')
                if len(parts) >= 6:
                    # Extract ADXL345 (cols 0,1,2) and ITG3200 (cols 3,4,5)
                    ax = float(parts[0]) * ACCEL_CONVERSION
                    ay = float(parts[1]) * ACCEL_CONVERSION
                    az = float(parts[2]) * ACCEL_CONVERSION
                    gx = float(parts[3]) * GYRO_CONVERSION
                    gy = float(parts[4]) * GYRO_CONVERSION
                    gz = float(parts[5]) * GYRO_CONVERSION
                    parsed_data.append([ax, ay, az, gx, gy, gz])
                    
            raw_array = np.array(parsed_data)
            
            # 3. Downsample from 200Hz to 50Hz (Take every 4th sample)
            # Ensure the array isn't empty before downsampling
            if raw_array.shape[0] > 0:
                downsampled_array = raw_array[::4, :]
                
                # 4. Chop into 2-second (100 sample) windows
                num_samples = downsampled_array.shape[0]
                for start_idx in range(0, num_samples - WINDOW_SIZE + 1, STEP_SIZE):
                    window = downsampled_array[start_idx : start_idx + WINDOW_SIZE, :]
                    all_windows.append(window)
                    all_labels.append(is_fall)
                
    # Convert lists to final numpy arrays
    X = np.array(all_windows)
    y = np.array(all_labels)
    
    print(f"Dataset processed! Found {len(X)} windows.")
    print(f"X shape: {X.shape} (Windows, Samples, Channels)")
    print(f"y shape: {y.shape} (Labels)")
    
    return X, y

if __name__ == "__main__":
    # Test the parser directly
    X, y = load_sisfall_data("./SisFall/SisFall_dataset")