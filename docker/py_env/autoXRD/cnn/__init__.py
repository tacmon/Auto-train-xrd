"""
PyTorch-based CNN module for XRD phase identification.

This module provides a pure PyTorch implementation for improved performance.
"""

import numpy as np
from random import shuffle

# PyTorch implementations
from .pytorch_models import (
    XRDNet, PDFNet, AlwaysDropout, XRDDataset, DataSetUp as PyTorchDataSetUp,
    train_model as pytorch_train_model, save_model, load_model,
    test_model as pytorch_test_model, main as pytorch_main
)
import torch

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)


class CustomDropout:
    """Legacy compatibility wrapper (no longer used)."""
    def __init__(self, rate, **kwargs):
        self.rate = rate
    def get_config(self):
        return {"rate": self.rate}


class DataSetUp(object):
    """Direct wrapper to PyTorch data setup."""

    def __init__(self, xrd, testing_fraction=0):
        """
        Args:
            xrd: a numpy array containing xrd spectra categorized by reference phase
            testing_fraction: fraction of data to reserve for testing
        """
        self.pytorch_setup = PyTorchDataSetUp(xrd, testing_fraction)
        self.xrd = xrd
        self.testing_fraction = testing_fraction
        self.num_phases = len(xrd)

    @property
    def phase_indices(self):
        """List of indices to keep track of xrd spectra."""
        return [v for v in range(self.num_phases)]

    @property
    def x(self):
        """Feature matrix (array of intensities used for training)"""
        intensities = []
        for (augmented_spectra, index) in zip(self.xrd, self.phase_indices):
            for pattern in augmented_spectra:
                intensities.append(pattern)
        return np.array(intensities)

    @property
    def y(self):
        """Target property to predict (one-hot encoded vectors)"""
        one_hot_vectors = []
        for (augmented_spectra, index) in zip(self.xrd, self.phase_indices):
            for pattern in augmented_spectra:
                assigned_vec = [0]*len(self.xrd)
                assigned_vec[index] = 1.0
                one_hot_vectors.append(assigned_vec)
        return np.array(one_hot_vectors)

    def split_training_testing(self):
        """Training and testing data split (legacy interface)"""
        x, y = self.x, self.y
        combined_xy = list(zip(x, y))
        shuffle(combined_xy)

        if self.testing_fraction == 0:
            train_x, train_y = zip(*combined_xy)
            return np.array(train_x), np.array(train_y), None, None
        else:
            total_samples = len(combined_xy)
            n_testing = int(self.testing_fraction*total_samples)
            train_xy = combined_xy[n_testing:]
            test_xy = combined_xy[:n_testing]
            train_x, train_y = zip(*train_xy)
            test_x, test_y = zip(*test_xy)
            return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def train_model(x_train, y_train, n_phases, num_epochs, is_pdf, n_dense=[3100, 1200], dropout_rate=0.7):
    """
    Train a PyTorch model (direct call to pytorch_models implementation).
    
    Args:
        x_train: numpy array of simulated xrd spectra
        y_train: one-hot encoded vectors associated with reference phase indices
        n_phases: number of reference phases considered
        num_epochs: number of training epochs
        is_pdf: whether to use PDF-optimized architecture
        n_dense: number of nodes comprising the two hidden layers
        dropout_rate: dropout probability for regularization
    Returns:
        model: trained PyTorch model
    """
    # Convert one-hot encoded labels to class indices
    y_class_indices = np.argmax(y_train, axis=1)
    
    # Reorganize data by class for PyTorch format
    temp_xrd_data = [[] for _ in range(n_phases)]
    for spectrum, class_idx in zip(x_train, y_class_indices):
        temp_xrd_data[class_idx].append(spectrum)
    temp_xrd_data = np.array(temp_xrd_data, dtype=object)
    
    # Use PyTorch implementation
    data_setup = PyTorchDataSetUp(temp_xrd_data, testing_fraction=0.0)
    train_loader, val_loader, _ = data_setup.get_dataloaders()
    
    model = pytorch_train_model(
        train_loader, val_loader, n_phases, num_epochs, is_pdf, n_dense, dropout_rate
    )
    return model


def test_model(model, test_x, test_y):
    """
    Test a PyTorch model.
    
    Args:
        model: trained PyTorch model
        test_x: feature matrix containing xrd spectra
        test_y: one-hot encoded vectors associated with the reference phases
    """
    y_class_indices = np.argmax(test_y, axis=1)
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(y_class_indices))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    acc = pytorch_test_model(model, test_loader)
    print('Test Accuracy: ' + str(acc) + '%')


def main(xrd, num_epochs, testing_fraction, is_pdf, fmodel='Model.pth'):
    """
    Main training function using PyTorch.
    
    Args:
        xrd: XRD/PDF spectra data
        num_epochs: Number of training epochs
        testing_fraction: Fraction of data for testing
        is_pdf: Whether to use PDF-optimized architecture
        fmodel: Filename to save the trained model
    """
    return pytorch_main(xrd, num_epochs, testing_fraction, is_pdf, fmodel)