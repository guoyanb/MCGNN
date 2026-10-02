# cuda_wavelet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CudaHaarWavelet(nn.Module):
    """GPU-accelerated Haar wavelet transform"""
    def __init__(self):
        super(CudaHaarWavelet, self).__init__()
        
        # Haar wavelet filter coefficients
        self.decomposition_filter = nn.Conv1d(
            in_channels=1, 
            out_channels=2, 
            kernel_size=2, 
            stride=2,
            bias=False,
            padding=0
        )
        
        # Initialize Haar wavelet filters
        self._init_haar_filters()
        
        # Freeze filter weights to preserve wavelet transform properties
        for param in self.decomposition_filter.parameters():
            param.requires_grad = False
    
    def _init_haar_filters(self):
        """Initialize Haar wavelet filters"""
        with torch.no_grad():
            # Haar low-pass filter (approximation coefficients): [1/√2, 1/√2]
            # Haar high-pass filter (detail coefficients): [1/√2, -1/√2]
            sqrt2 = np.sqrt(2)
            haar_weights = torch.tensor([
                [[1/sqrt2, 1/sqrt2]],   # Low-pass filter
                [[1/sqrt2, -1/sqrt2]]   # High-pass filter
            ], dtype=torch.float32)
            
            self.decomposition_filter.weight.data = haar_weights
    
    def forward(self, x):
        """
        Forward pass: perform 1D Haar wavelet transform
        Args:
            x: Input signal, shape [batch_size, signal_length]
        Returns:
            coeffs: List of wavelet coefficients, each element of shape [batch_size, coeff_length]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # If 1D, add batch dimension
        elif x.dim() > 2:
            x = x.view(x.shape[0], -1)  # Flatten extra dimensions
        
        batch_size, signal_len = x.shape
        
        # Ensure signal length is even
        if signal_len % 2 != 0:
            padding = torch.zeros(batch_size, 1, device=x.device)
            x = torch.cat([x, padding], dim=1)
            signal_len += 1
        
        # Reshape to shape required by conv layer [batch, channels, length]
        x = x.unsqueeze(1)  # [batch, 1, signal_len]
        
        # Perform wavelet transform
        coeffs = self.decomposition_filter(x)  # [batch, 2, signal_len//2]
        
        # Separate approximation and detail coefficients
        approx = coeffs[:, 0, :]  # Approximation coefficients (low frequency)
        detail = coeffs[:, 1, :]  # Detail coefficients (high frequency)
        
        return approx, detail


class MultiLevelCudaWavelet(nn.Module):
    """GPU-accelerated multi-level wavelet decomposition"""
    def __init__(self, wavelet_type='haar', levels=3):
        super(MultiLevelCudaWavelet, self).__init__()
        
        self.wavelet_type = wavelet_type
        self.levels = levels
        
        if wavelet_type.lower() == 'haar':
            self.wavelet = CudaHaarWavelet()
        else:
            raise ValueError(f"Unsupported wavelet type: {wavelet_type}")
        
    def decompose(self, x, level=None):
        """
        Multi-level wavelet decomposition
        Args:
            x: Input signal [batch_size, signal_length]
            level: Decomposition level; if None, use self.levels
        Returns:
            coeffs_list: Coefficient list, [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        """
        if level is None:
            level = self.levels
        
        coeffs_list = []
        current_signal = x.clone()
        
        for i in range(level):
            # If current signal length is less than 2, stop decomposition
            if current_signal.shape[1] < 2:
                coeffs_list.append(current_signal)
                break
            
            # Perform one-level decomposition
            approx, detail = self.wavelet(current_signal)
            
            # Save detail coefficients
            coeffs_list.insert(0, detail)  # Insert at the beginning of the list
            
            # Prepare for next-level decomposition
            current_signal = approx
            
            # If last level is reached, save approximation coefficients
            if i == level - 1:
                coeffs_list.insert(0, approx)  # Approximation coefficients at the front
        
        return coeffs_list
    
    def forward(self, x, level=None):
        """Perform multi-level wavelet decomposition and return flattened coefficients"""
        coeffs = self.decompose(x, level)
        
        # Flatten all coefficients
        flattened_coeffs = []
        for coeff in coeffs:
            flattened_coeffs.append(coeff.view(coeff.shape[0], -1))
        
        # Concatenate all coefficients
        if flattened_coeffs:
            return torch.cat(flattened_coeffs, dim=1)
        else:
            return torch.zeros(x.shape[0], 0, device=x.device)


class CudaWaveletTransform(nn.Module):
    """GPU-accelerated wavelet transform - simplified version for feature extraction only"""
    def __init__(self, wavelet_type='haar', levels=3, hidden_size=64, device='cpu'):
        super(CudaWaveletTransform, self).__init__()
        
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.hidden_size = hidden_size
        self.device = device
        
        # Multi-level wavelet decomposition
        self.multi_level_wavelet = MultiLevelCudaWavelet(wavelet_type, levels)
        
        # Adaptive pooling layer to unify coefficients of different lengths to a fixed dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(hidden_size)
    
    def _preprocess_signal(self, x):
        """Preprocess input signal"""
        original_shape = x.shape
        
        # Flatten processing
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])
        
        # Ensure signal length is sufficient
        min_length = 2 ** self.levels
        if x.shape[1] < min_length:
            # If signal is too short, perform zero padding
            padding_size = min_length - x.shape[1]
            padding = torch.zeros(x.shape[0], padding_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        return x, original_shape
    
    def forward(self, x):
        """GPU-accelerated wavelet feature extraction"""
        # 1. Preprocessing
        x_processed, original_shape = self._preprocess_signal(x)
        
        # 2. Perform multi-level wavelet decomposition
        wavelet_coeffs = self.multi_level_wavelet(x_processed, self.levels)
        
        # 3. Adaptive pooling to fixed dimension
        if wavelet_coeffs.shape[1] > 0:
            # Reshape to [batch, 1, length] for 1D pooling
            coeffs_reshaped = wavelet_coeffs.unsqueeze(1)
            pooled_coeffs = self.adaptive_pool(coeffs_reshaped)
            pooled_coeffs = pooled_coeffs.squeeze(1)
        else:
            pooled_coeffs = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        
        # 4. Restore original shape (if needed)
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (self.hidden_size,)
            pooled_coeffs = pooled_coeffs.view(new_shape)
        
        return pooled_coeffs


