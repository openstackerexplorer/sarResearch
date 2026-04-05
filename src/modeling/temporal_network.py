import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

class ConvLSTMCell(nn.Module):
    """
    A single Convolutional LSTM Cell to process spatial-temporal SAR data.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim, # For the 4 LSTM gates (i, f, o, g)
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # Concatenate spatial image input and previous hidden state along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # LSTM Gate Math
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class SARAnomalyDetector(nn.Module):
    """
    The full temporal anomaly detection model.
    Takes a temporal stack of InSAR phase/amplitude data and outputs a 2D spatial anomaly map.
    """
    def __init__(self, input_channels=2, hidden_channels=16, kernel_size=(3,3)):
        super(SARAnomalyDetector, self).__init__()
        
        # We assume 2 input channels: Amplitude and Phase (or Real and Imaginary)
        self.convlstm = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        
        # A final 1x1 convolution to collapse the hidden states into a single 2D anomaly score map
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Time_Steps, Channels, Height, Width)
        b, t, c, h, w = x.size()
        
        # Initialize hidden state and cell state with zeros
        h_state = torch.zeros(b, self.convlstm.hidden_dim, h, w, device=x.device)
        c_state = torch.zeros(b, self.convlstm.hidden_dim, h, w, device=x.device)
        
        # Pass the temporal sequence through the ConvLSTM step by step
        for time_step in range(t):
            h_state, c_state = self.convlstm(x[:, time_step, :, :, :], (h_state, c_state))
            
        # Analyze the final hidden state to determine structural anomalies
        anomaly_map = self.final_conv(h_state)
        return anomaly_map # Shape: (Batch, 1, Height, Width)


# --- Pipeline Integration Wrapper ---

def detect_structural_anomalies(sar_stack_array: np.ndarray, osm_masks: dict) -> List[Dict]:
    """
    Wraps the PyTorch model for the LangGraph processing_node.
    
    Args:
        sar_stack_array: A 4D numpy array (Time, Channels, Height, Width).
                         Channels should be calibrated Amplitude and Phase.
        osm_masks: Dictionary mapping infrastructure names to 2D boolean pixel masks.
        
    Returns:
        List of dictionaries formatted for the LLM Assessment Node.
    """
    print("Initializing PyTorch SAR Anomaly Detector...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model (In a real scenario, you would load pre-trained weights here)
    model = SARAnomalyDetector(input_channels=2, hidden_channels=16).to(device)

    # 2. LOAD THE TRAINED WEIGHTS (This is what was missing!)
    weights_path = "sar_model_weights.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded pre-trained model weights.")
    except FileNotFoundError:
        print("WARNING: sar_model_weights.pth not found. Running with random initialization!")
    
    # 3. Set to Inference mode

    model.eval() # Set to inference mode
    
    # Prepare the input tensor (Add a Batch dimension of 1)
    # Shape becomes (1, Time, Channels, Height, Width)
    input_tensor = torch.tensor(sar_stack_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    print(f"Running ConvLSTM inference on temporal stack of shape: {input_tensor.shape}...")
    with torch.no_grad():
        # Output shape is (1, 1, Height, Width)
        anomaly_map_tensor = model(input_tensor)
        
    # Convert back to a 2D numpy array for spatial parsing
    anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()
    
    detected_anomalies = []
    
    # Iterate over the OpenStreetMap footprints we projected earlier
    for asset_name, mask_array in osm_masks.items():
        # Extract the anomaly scores only for the pixels inside this specific building
        building_pixels = anomaly_map[mask_array]
        
        if len(building_pixels) == 0:
            continue
            
        # Calculate the average displacement/anomaly score for the foundation
        avg_score = float(np.mean(building_pixels))

        # 🚨 ADD THIS DEBUG PRINT 🚨
        print(f"🔬 DEBUG - {asset_name} | Raw NN Output Score: {avg_score:.6f}")
        
        # If the model detects a significant deviation (e.g., threshold > 0.7)
        if abs(avg_score) > 0.01:
            # Convert the arbitrary neural network score to estimated millimeters 
            # (Requires phase-unwrapping/wavelength math in a full implementation)
            estimated_mm = avg_score * -250.0 
            
            detected_anomalies.append({
                "asset_type": "Mapped Infrastructure", # We would pull this from OSM properties
                "name": asset_name,
                "subsidence_mm": round(estimated_mm, 2),
                "confidence_score": round(abs(avg_score), 2),
                "temporal_trend": "detected_deformation"
            })
            
    return detected_anomalies