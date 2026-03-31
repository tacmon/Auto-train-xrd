"""
PyTorch-based inference utilities for spectrum analysis.

This module provides PyTorch implementations of the inference components
previously dependent on TensorFlow.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List
import os
import warnings

def load_pytorch_model(model_path: str, device: str = None):
    """
    Load a PyTorch model for inference.
    
    Args:
        model_path: Path to the saved PyTorch model (.pth file)
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    if not model_path.endswith('.pth'):
        raise ValueError(f"Only PyTorch models (.pth) are supported. Got: {model_path}")
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    from autoXRD.cnn.pytorch_models import load_model
    return load_model(model_path, device)


class PyTorchDropoutPrediction:
    """
    PyTorch implementation of Monte Carlo dropout for uncertainty estimation.
    
    Replaces the TensorFlow-based KerasDropoutPrediction with improved performance
    and better uncertainty quantification.
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained PyTorch model with AlwaysDropout layers
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def predict(self, x: np.ndarray, min_conf: float = 10.0, n_iter: int = 100) -> Tuple[np.ndarray, int, List[float], int]:
        """
        Perform Monte Carlo dropout prediction for uncertainty estimation.
        
        Args:
            x: Input spectrum to be classified
            min_conf: Minimum confidence threshold (as percentage)
            n_iter: Number of Monte Carlo iterations
            
        Returns:
            Tuple containing:
            - prediction: Mean prediction probabilities across all iterations
            - num_phases: Number of phases with confidence above threshold
            - certainties: List of confidence values for qualifying phases
            - num_outputs: Total number of possible output classes
        """
        # Convert from % to 0-1 fractional
        if min_conf > 1.0:
            min_conf /= 100.0
            
        # Ensure model is in eval mode but dropout layers will still be active
        self.model.eval()
        
        # Convert input to tensor
        if isinstance(x, (list, np.ndarray)):
            x = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
        x = x.to(self.device)
        
        # Ensure correct input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 3 and x.shape[2] == 1:
            x = x.transpose(1, 2)  # (batch, length, 1) -> (batch, 1, length)
            
        # Monte Carlo Dropout iterations
        results = []
        with torch.no_grad():
            for _ in range(n_iter):
                output = self.model(x)
                results.append(output.cpu().numpy().flatten())
        
        results = np.array(results)  # Shape: (n_iter, num_classes)
        prediction = results.mean(axis=0)  # Average prediction
        num_outputs = len(prediction)
        
        # Calculate phase certainties based on prediction consistency
        all_preds = [np.argmax(pred) for pred in results]  # Most likely class for each iteration
        
        # Count occurrences of each predicted class
        unique_classes, counts = np.unique(all_preds, return_counts=True)
        
        # Calculate confidence as frequency of prediction
        certainties = []
        for count in counts:
            conf = count / n_iter
            if conf >= min_conf:
                certainties.append(conf)
        
        # Sort certainties in descending order
        certainties = sorted(certainties, reverse=True)
        
        return prediction, len(certainties), certainties, num_outputs


def ensure_pytorch_model(model_path: str, reference_phases: List[str] = None, is_pdf: bool = False):
    """
    Ensure a PyTorch model exists.
    
    Args:
        model_path: Path to the model file (must be .pth)
        reference_phases: List of reference phase names (unused, kept for compatibility)
        is_pdf: Whether this is a PDF model (unused, kept for compatibility)
        
    Returns:
        Path to the PyTorch model
    """
    if not model_path.endswith('.pth'):
        raise ValueError(f"Only PyTorch models (.pth) are supported. Got: {model_path}")
    
    if os.path.exists(model_path):
        return model_path
    else:
        raise FileNotFoundError(f"PyTorch model not found: {model_path}")


class ModelLoader:
    """
    Utility class for loading and managing PyTorch models for inference.
    """
    
    def __init__(self):
        self._models = {}  # Cache for loaded models
        
    def load_model(self, model_path: str, reference_phases: List[str] = None, 
                   is_pdf: bool = False, device: str = None):
        """
        Load a PyTorch model for inference, with caching.
        
        Args:
            model_path: Path to the PyTorch model file (.pth)
            reference_phases: List of reference phase names (unused, kept for compatibility)
            is_pdf: Whether this is a PDF model (unused, kept for compatibility)
            device: Device to load the model on
            
        Returns:
            Loaded PyTorch model
        """
        # Check cache first
        cache_key = (model_path, device)
        if cache_key in self._models:
            return self._models[cache_key]
            
        # Ensure we have a valid PyTorch model
        pytorch_path = ensure_pytorch_model(model_path, reference_phases, is_pdf)
        
        # Load the model
        model = load_pytorch_model(pytorch_path, device)
        
        # Cache the model
        self._models[cache_key] = model
        
        return model
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        self._models.clear()


# Global model loader instance
_model_loader = ModelLoader()

def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return _model_loader