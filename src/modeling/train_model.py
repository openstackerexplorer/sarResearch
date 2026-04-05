import torch
import glob,os,sys
import torch.nn as nn
import torch.optim as optim
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# Add the project root to Python's system path before doing any absolute imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from src.sar_processing.slc_utils import load_and_calibrate_slc
# Import the network architecture we already built
# from temporal_network import SARAnomalyDetector

from src.modeling.temporal_network import SARAnomalyDetector
from src.sar_processing.slc_utils import load_and_calibrate_slc

def train_predictive_autoencoder(dummy_training_data: np.ndarray, epochs=50):
    """
    Trains the ConvLSTM to predict the NEXT radar frame based on the previous frames.
    This teaches the model what 'normal' infrastructure stability looks like.
    """
    print("Initializing Offline Training on Stable Infrastructure...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = SARAnomalyDetector(input_channels=2, hidden_channels=16).to(device)
    
    # We use Mean Squared Error to measure how badly it predicts the next frame
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train() # Set to training mode
    
    # In a real scenario, this is a PyTorch DataLoader yielding batches of SAR stacks
    # Shape: (Batch, Time, Channels, Height, Width)
    inputs = torch.tensor(dummy_training_data, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass: the model tries to predict the spatial phase
        reconstruction = model(inputs)
        
        # Loss: How far off was the model's prediction from a stable '0' deviation?
        # (Assuming we normalize stable ground to 0)
        target = torch.zeros_like(reconstruction)
        loss = criterion(reconstruction, target)
        
        # Backward pass & optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {loss.item():.6f}")

    # SAVE THE WEIGHTS!
    save_path = "sar_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}")


def build_real_training_stack(data_dir, patch_size=256):
    """
    Loads your downloaded Capella data, crops it to a manageable patch size to save RAM,
    and builds the temporal stack for training.
    """
    print(f"Loading real SAR data from disk: {data_dir}")
    tif_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {data_dir}.")
        
    stack = []
    valid_pairs = 0
    
    for tif in tif_files:
        base_name = os.path.basename(tif)
        parts = base_name.split('_')
        stac_id = "_".join(parts[:7]) if len(parts) >= 7 else base_name.split('.')[0]
            
        matching_jsons = glob.glob(os.path.join(data_dir, f"*{stac_id}*.json"))
        target_jsons = [j for j in matching_jsons if 'extended' in j.lower()]
        if not target_jsons and matching_jsons:
            target_jsons = matching_jsons
            
        if not target_jsons:
            print(f"Warning: No matching JSON found for {base_name}. Skipping.")
            continue
            
        json_meta = target_jsons[0]
        
        try:
            # Load and calibrate the complex array
            calibrated_slc = load_and_calibrate_slc(tif, json_meta)
            
            # --- THE FIX: CROP THE IMAGE TO SAVE RAM ---
            h, w = calibrated_slc.shape
            
            # Calculate the center coordinates
            center_y, center_x = h // 2, w // 2
            
            # Define the bounding box for our patch
            half_patch = patch_size // 2
            start_y = max(0, center_y - half_patch)
            end_y = start_y + patch_size
            start_x = max(0, center_x - half_patch)
            end_x = start_x + patch_size
            
            # Slice the massive array down to just 256x256 pixels!
            cropped_slc = calibrated_slc[start_y:end_y, start_x:end_x]
            # ---------------------------------------------
            
            # Split into Real and Imaginary using the CROPPED array
            real_channel = np.real(cropped_slc)
            imag_channel = np.imag(cropped_slc)
            
            # Combine into (2, Height, Width)
            frame = np.stack([real_channel, imag_channel], axis=0)
            stack.append(frame)
            valid_pairs += 1
            print(f"Successfully loaded, cropped, and paired: {base_name}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            
    if valid_pairs < 2:
        raise ValueError(f"Need at least 2 valid pairs. Found {valid_pairs}.")
        
    # Convert list to numpy array of shape (Time, Channels, Height, Width)
    stack_array = np.array(stack)
    print(f"\nSuccessfully built temporal stack! Final array size in memory: {stack_array.nbytes / (1024*1024):.2f} MB")
    
    # Add the Batch dimension
    return np.expand_dims(stack_array, axis=0)
# if __name__ == "__main__":
#     # Create some dummy data (1 batch, 10 time steps, 2 channels, 256x256 pixels)
#     print("Generating synthetic stable SAR stack for training...")
#     dummy_data = np.random.randn(1, 10, 2, 256, 256) * 0.1 
    
#     train_predictive_autoencoder(dummy_data, epochs=50)

if __name__ == "__main__":
    # 1. Load the REAL data you downloaded via the LangGraph agent!
    RAW_DATA_DIR = "data/raw/stack_40"
    
    try:
        real_training_data = build_real_training_stack(RAW_DATA_DIR)
        print(f"Successfully loaded real data shape: {real_training_data.shape}")
        
        # 2. Train the model on the real physics of San Jose
        train_predictive_autoencoder(real_training_data, epochs=50)
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Please ensure your LangGraph agent has successfully downloaded the files first.")