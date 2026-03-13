import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class ConvLSTMCell(nn.Module):
    """
    Standard Convolutional LSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class TemporalAnomalyDetector(nn.Module):
    """
    ConvLSTM-based network for detecting structural anomalies in temporal SAR sequences.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(TemporalAnomalyDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=True)
        self.anomaly_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sequence of spatial features (time, channels, h, w).
        """
        # x: (T, C, H, W) -> we need (B, T, C, H, W)
        x = x.unsqueeze(0) 
        b, t, c, h, w = x.size()
        
        h_cur = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        c_cur = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        
        for time_step in range(t):
            h_cur, c_cur = self.cell(x[:, time_step, :, :, :], (h_cur, c_cur))
            
        return self.anomaly_head(h_cur)

def build_convlstm_model(input_shape: Tuple[int, int, int]) -> TemporalAnomalyDetector:
    """Constructs the Convolutional LSTM network."""
    # input_shape: (channels, h, w)
    return TemporalAnomalyDetector(input_dim=input_shape[0], hidden_dim=64, kernel_size=3)

def detect_structural_anomalies(model: TemporalAnomalyDetector, temporal_features: np.ndarray) -> np.ndarray:
    """
    Analyzes the time-series features for anomalies.
    
    Args:
        model: TemporalAnomalyDetector instance.
        temporal_features: Array of spatial features (time, channels, h, w).
        
    Returns:
        Anomaly heatmap.
    """
    x = torch.from_numpy(temporal_features).float()
    with torch.no_grad():
        anomaly_map = model(x)
    return anomaly_map.squeeze().numpy()
