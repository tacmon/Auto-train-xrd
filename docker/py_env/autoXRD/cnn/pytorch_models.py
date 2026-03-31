"""
PyTorch implementation of XRD/PDF neural networks for phase identification.

This module provides a complete replacement for the TensorFlow-based implementation,
offering improved performance, better uncertainty estimation, and enhanced training capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from random import shuffle
from typing import Tuple, List, Optional
import sys
import os
from tqdm import tqdm
from typing import List

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)


class AlwaysDropout(nn.Module):
    """
    Custom dropout layer that applies dropout during both training and inference.
    This is essential for Monte Carlo dropout uncertainty estimation.
    """
    
    def __init__(self, p: float = 0.5):
        super(AlwaysDropout, self).__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode."""
        return F.dropout(x, p=self.p, training=True)

# ---------------------------
# DynamicConv1D
# ---------------------------
class DynamicConv1D(nn.Module):
    """
    Dynamic convolutional layer that dynamically adjusts the kernel size based on input size.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels * kernel_size)
        )

        # Initialize weights to zero, not disrupt backbone stability
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        pooled = self.global_pool(x).squeeze(-1)          # [B, C]
        weights = self.mlp(pooled)                        # [B, C*kernel]
        weights = weights.view(B * C, 1, self.kernel_size)

        x_ = x.reshape(1, B * C, L)
        out = F.conv1d(x_, weights, padding=self.kernel_size // 2, groups=B * C)
        out = out.reshape(B, C, L)
        return out

# ---------------------------
# XRDNet with DynamicConv1D
# ---------------------------
class XRDNetWithDynamic(nn.Module):
    """
    XRDNet with DynamicConv1D for XRD pattern classification.
    """
    def __init__(self, num_classes: int, n_dense: List[int] = [3100, 1200], dropout_rate: float = 0.7):
        super().__init__()

        # Main branch convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=1, padding=35 // 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # New branches: large kernel and dynamic convolution
        self.large_kernel = nn.Conv1d(64, 64, kernel_size=101, stride=1, padding=101 // 2)
        self.dynamic = DynamicConv1D(64, kernel_size=3)

        # Subsequent convolutional layers
        self.conv2 = nn.Conv1d(64, 64, kernel_size=30, stride=1, padding=30 // 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=25, stride=1, padding=25 // 2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=20, stride=1, padding=20 // 2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(64, 64, kernel_size=15, stride=1, padding=15 // 2)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=10 // 2)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        # flatten
        self._calculate_conv_output_size()

        # Full connection layers
        self.dropout1 = AlwaysDropout(dropout_rate)
        self.fc1 = nn.Linear(self.conv_output_size, n_dense[0])
        self.bn1 = nn.BatchNorm1d(n_dense[0])

        self.dropout2 = AlwaysDropout(dropout_rate)
        self.fc2 = nn.Linear(n_dense[0], n_dense[1])
        self.bn2 = nn.BatchNorm1d(n_dense[1])

        self.dropout3 = AlwaysDropout(dropout_rate)
        self.fc3 = nn.Linear(n_dense[1], num_classes)

    def _calculate_conv_output_size(self):
        """
        Calculate the output size dynamically using a dummy forward pass.
        This approach is completely robust to any changes in kernel_size, stride, or padding.
        """
        # 1. 创建一个与实际输入形状匹配的假张量 (Batch_Size=1, Channels=1, Length=4501)
        # 注意：这个设备默认在 CPU 上。因为此函数通常在 __init__ 中调用，此时模型也还在 CPU 上。
        dummy_input = torch.zeros(1, 1, 4501)
        
        # 2. 临时关闭梯度计算（节省内存并加速前向传播）
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.pool2(x)
            
            x = self.conv3(x)
            x = self.pool3(x)
            
            x = self.conv4(x)
            x = self.pool4(x)
            
            x = self.conv5(x)
            x = self.pool5(x)
            
            x = self.conv6(x)
            x = self.pool6(x)
            
        # 3. x.numel() 会返回张量中元素的总个数（即 64 通道 * 71 长度 = 4544）
        # 这个值将直接用于后续全连接层 nn.Linear(self.conv_output_size, ...) 的 in_features
        self.conv_output_size = x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # [B, L]
            x = x.unsqueeze(1)
        if x.dim() == 3 and x.shape[1] == 1:  # [B,1,L]
            pass

        # conv1 + pool1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # --- Fuse large kernel and dynamic convolution ---
        x_lk = self.large_kernel(x)   # Static large kernel
        x_dyn = self.dynamic(x)       # Dynamic small kernel
        x = x + x_lk + x_dyn

        # Subsequent convolutional layers
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)
        x = F.relu(self.conv4(x)); x = self.pool4(x)
        x = F.relu(self.conv5(x)); x = self.pool5(x)
        x = F.relu(self.conv6(x)); x = self.pool6(x)

        # flatten + fully connected layers
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x)); x = self.bn1(x)

        x = self.dropout2(x)
        x = F.relu(self.fc2(x)); x = self.bn2(x)

        x = self.dropout3(x)
        x = self.fc3(x)

        return F.softmax(x, dim=1)
    
class XRDNet(nn.Module):
    """
    1D Convolutional Neural Network for XRD pattern classification.
    
    Optimized architecture for XRD analysis with progressive kernel size reduction
    and custom dropout for uncertainty estimation.
    """
    
    def __init__(self, num_classes: int, n_dense: List[int] = [3100, 1200], 
                 dropout_rate: float = 0.7):
        """
        Args:
            num_classes: Number of reference phases (output classes)
            n_dense: List containing sizes of dense layers [layer1_size, layer2_size]
            dropout_rate: Dropout probability for regularization
        """
        super(XRDNet, self).__init__()
        
        # Convolutional layers with progressive kernel size reduction and optimized pooling
        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=1, padding=35//2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=30, stride=1, padding=30//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=25, stride=1, padding=25//2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=20, stride=1, padding=20//2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv1d(64, 64, kernel_size=15, stride=1, padding=15//2)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=10//2)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        
        # Calculate flattened size after all conv and pooling layers
        self._calculate_conv_output_size()
        
        # Dense layers with dropout and batch normalization
        self.dropout1 = AlwaysDropout(dropout_rate)
        self.fc1 = nn.Linear(self.conv_output_size, n_dense[0])
        self.bn1 = nn.BatchNorm1d(n_dense[0])
        
        self.dropout2 = AlwaysDropout(dropout_rate)
        self.fc2 = nn.Linear(n_dense[0], n_dense[1])
        self.bn2 = nn.BatchNorm1d(n_dense[1])
        
        self.dropout3 = AlwaysDropout(dropout_rate)
        self.fc3 = nn.Linear(n_dense[1], num_classes)
        
    def _calculate_conv_output_size(self):
        """
        Calculate the output size dynamically using a dummy forward pass.
        This approach is completely robust to any changes in kernel_size, stride, or padding.
        """
        # 1. 创建一个与实际输入形状匹配的假张量 (Batch_Size=1, Channels=1, Length=4501)
        # 注意：这个设备默认在 CPU 上。因为此函数通常在 __init__ 中调用，此时模型也还在 CPU 上。
        dummy_input = torch.zeros(1, 1, 4501)
        
        # 2. 临时关闭梯度计算（节省内存并加速前向传播）
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.pool2(x)
            
            x = self.conv3(x)
            x = self.pool3(x)
            
            x = self.conv4(x)
            x = self.pool4(x)
            
            x = self.conv5(x)
            x = self.pool5(x)
            
            x = self.conv6(x)
            x = self.pool6(x)
            
        # 3. x.numel() 会返回张量中元素的总个数（即 64 通道 * 71 长度 = 4544）
        # 这个值将直接用于后续全连接层 nn.Linear(self.conv_output_size, ...) 的 in_features
        self.conv_output_size = x.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4501, 1) or (batch_size, 1, 4501)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Ensure correct input shape: (batch_size, channels, length)
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.transpose(1, 2)  # (batch_size, 4501, 1) -> (batch_size, 1, 4501)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 4501) -> (batch_size, 1, 4501)
            
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        
        # Flatten for dense layers
        x = x.view(x.size(0), -1)
        
        # Dense layers with dropout and batch normalization
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)


class PDFNet(nn.Module):
    """
    Simplified 1D CNN optimized for PDF (Pair Distribution Function) analysis.
    
    Uses a single large convolutional kernel followed by aggressive pooling,
    designed specifically for PDF pattern characteristics.
    """
    
    def __init__(self, num_classes: int, n_dense: List[int] = [3100, 1200], 
                 dropout_rate: float = 0.7):
        """
        Args:
            num_classes: Number of reference phases (output classes)
            n_dense: List containing sizes of dense layers [layer1_size, layer2_size]
            dropout_rate: Dropout probability for regularization
        """
        super(PDFNet, self).__init__()
        
        # Single convolutional layer with large kernel
        self.conv1 = nn.Conv1d(1, 64, kernel_size=60, stride=1, padding='same')
        
        # Aggressive pooling sequence
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        # Calculate flattened size
        self._calculate_conv_output_size()
        
        # Dense layers
        self.dropout1 = AlwaysDropout(dropout_rate)
        self.fc1 = nn.Linear(self.conv_output_size, n_dense[0])
        self.bn1 = nn.BatchNorm1d(n_dense[0])
        
        self.dropout2 = AlwaysDropout(dropout_rate)
        self.fc2 = nn.Linear(n_dense[0], n_dense[1])
        self.bn2 = nn.BatchNorm1d(n_dense[1])
        
        self.dropout3 = AlwaysDropout(dropout_rate)
        self.fc3 = nn.Linear(n_dense[1], num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate the output size after all pooling layers."""
        size = 4501
        size = ((size + 2*1 - 3) // 2) + 1  # pool1
        size = ((size + 2*1 - 3) // 2) + 1  # pool2
        size = ((size + 2*0 - 2) // 2) + 1  # pool3
        size = ((size + 2*0 - 1) // 2) + 1  # pool4
        size = ((size + 2*0 - 1) // 2) + 1  # pool5
        size = ((size + 2*0 - 1) // 2) + 1  # pool6
        self.conv_output_size = size * 64
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PDF network."""
        # Ensure correct input shape
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Convolutional layer
        x = F.relu(self.conv1(x))
        
        # Pooling sequence
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = self.pool6(x)
        
        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)


class XRDDataset(Dataset):
    """
    Custom PyTorch Dataset for XRD/PDF spectra data.
    
    Handles the conversion from the original nested numpy array format
    to a format suitable for PyTorch DataLoader.
    """
    
    def __init__(self, xrd_data: np.ndarray):
        """
        Args:
            xrd_data: Numpy array of shape (num_phases, num_spectra_per_phase, 4501, 1)
        """
        self.spectra = []
        self.labels = []
        
        # Flatten the nested structure and create labels
        for phase_idx, phase_spectra in enumerate(xrd_data):
            for spectrum in phase_spectra:
                # Convert to tensor and ensure correct shape
                spectrum_tensor = torch.FloatTensor(spectrum).squeeze()  # Remove extra dims
                if spectrum_tensor.dim() == 0:
                    spectrum_tensor = spectrum_tensor.unsqueeze(0)
                    
                self.spectra.append(spectrum_tensor)
                self.labels.append(phase_idx)
        
        self.spectra = torch.stack(self.spectra)
        self.labels = torch.LongTensor(self.labels)
        
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx], self.labels[idx]


class DataSetUp(object):
    """
    Enhanced data setup class for PyTorch training.
    
    Provides train/test splitting and DataLoader creation with improved
    functionality compared to the original TensorFlow version.
    """
    
    def __init__(self, xrd: np.ndarray, testing_fraction: float = 0.0, 
                 batch_size: int = 32, num_workers: int = 4):
        """
        Args:
            xrd: Numpy array containing XRD spectra categorized by reference phase
            testing_fraction: Fraction of data to reserve for testing
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for data loading
        """
        self.xrd = xrd
        self.testing_fraction = testing_fraction
        self.num_phases = len(xrd)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create train, validation, and optionally test DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
            test_loader is None if testing_fraction is 0
        """
        # Create dataset
        dataset = XRDDataset(self.xrd)
        
        if self.testing_fraction == 0:
            # Split into train (80%) and validation (20%)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(1)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, None
            
        else:
            # Split into train/val and test
            test_size = int(self.testing_fraction * len(dataset))
            trainval_size = len(dataset) - test_size
            
            trainval_dataset, test_dataset = random_split(
                dataset, [trainval_size, test_size],
                generator=torch.Generator().manual_seed(1)
            )
            
            # Further split train/val
            train_size = int(0.8 * trainval_size)
            val_size = trainval_size - train_size
            
            train_dataset, val_dataset = random_split(
                trainval_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(1)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, test_loader

def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                num_phases: int, num_epochs: int, is_pdf: bool, 
                n_dense: List[int] = [3100, 1200], dropout_rate: float = 0.7,
                learning_rate: float = 0.0005, device: str = None,
                patience: int = 15, lr_patience: int = 8,
                use_dynamic: bool = False) -> nn.Module:

    """
    Train the neural network model with enhanced features including learning rate scheduling.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        num_phases: Number of reference phases
        num_epochs: Number of training epochs
        is_pdf: Whether to use PDF-optimized architecture
        n_dense: Dense layer sizes
        dropout_rate: Dropout probability
        learning_rate: Initial learning rate for optimizer (reduced from 0.001 to 0.0005)
        device: Device to use for training ('cuda' or 'cpu')
        patience: Early stopping patience (epochs without improvement)
        lr_patience: Learning rate scheduler patience (epochs without improvement before reducing LR)
        
    Returns:
        Trained model
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    print(f"Training on device: {device}")
    

    if is_pdf:
        model = PDFNet(num_phases, n_dense, dropout_rate)
        print("Using PDF-optimized architecture")
    else:
        if use_dynamic:
            model = XRDNetWithDynamic(num_phases, n_dense, dropout_rate)
            print("Using XRDNet with DynamicConv1D")
        else:
            model = XRDNet(num_phases, n_dense, dropout_rate)
            print("Using standard XRDNet")

        
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=lr_patience, min_lr=1e-6
    )
    
    # Early stopping variables
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Training loop with progress tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                epoch_val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping and best model tracking
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ New best validation accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%, "
              f"LR: {current_lr:.2e}, "
              f"Patience: {patience_counter}/{patience}")
    
    # Load best model if we have one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model with validation accuracy: {best_val_accuracy:.2f}%")
    
    print("Training completed!")
    print(f"Final validation accuracy: {max(val_accuracies):.2f}%")
    
    return model

def save_model(model: nn.Module, filepath: str, is_pdf: bool = False, use_dynamic: bool = False):
    """
    Save the trained model with metadata.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    model_type = 'PDFNet' if is_pdf else ('XRDNetWithDynamic' if use_dynamic else 'XRDNet')

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'num_classes': model.fc3.out_features,
        'n_dense': [model.fc1.out_features, model.fc2.out_features],
        'dropout_rate': model.dropout1.p if hasattr(model, "dropout1") else None,
        'is_pdf': is_pdf,
        'use_dynamic': use_dynamic
    }, filepath)

    print(f"Model saved to: {filepath}")
    print(f"Model type: {model_type}")

def load_model(filepath: str, device: str = None) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    checkpoint = torch.load(filepath, map_location=device)

    is_pdf = checkpoint.get('is_pdf', False)
    use_dynamic = checkpoint.get('use_dynamic', False)

    if is_pdf:
        model = PDFNet(
            checkpoint['num_classes'],
            checkpoint['n_dense'],
            checkpoint['dropout_rate']
        )
    else:
        if use_dynamic:
            model = XRDNetWithDynamic(
                checkpoint['num_classes'],
                checkpoint['n_dense'],
                checkpoint['dropout_rate']
            )
        else:
            model = XRDNet(
                checkpoint['num_classes'],
                checkpoint['n_dense'],
                checkpoint['dropout_rate']
            )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded from: {filepath}")
    print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
    print(f"Number of classes: {checkpoint['num_classes']}")
    return model

def main(xrd: np.ndarray, num_epochs: int, testing_fraction: float, 
         is_pdf: bool, fmodel: str = 'Model.pth',
         use_dynamic: bool = False):
    """
    Main training function with enhanced PyTorch implementation.
    
    Args:
        xrd: XRD/PDF spectra data
        num_epochs: Number of training epochs
        testing_fraction: Fraction of data for testing
        is_pdf: Whether to use PDF-optimized architecture
        fmodel: Filename to save the trained model
        use_dynamic: Whether to use XRDNetWithDynamic instead of XRDNet
    """
    print("Setting up data...")
    data_setup = DataSetUp(xrd, testing_fraction)
    num_phases = data_setup.num_phases
    
    print(f"Number of phases: {num_phases}")
    print(f"Dataset size: {len(XRDDataset(xrd))} spectra")
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_setup.get_dataloaders()
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Train model
    model = train_model(
        train_loader, val_loader,
        num_phases=num_phases,
        num_epochs=num_epochs,
        is_pdf=is_pdf,
        use_dynamic=use_dynamic
    )
    
    # Save model with extra flag
    save_model(model, fmodel, is_pdf=is_pdf, use_dynamic=use_dynamic)
    
    # Test model if test data is available
    if test_loader is not None:
        test_accuracy = test_model(model, test_loader)
        print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    return model

def test_model(model: nn.Module, test_loader: DataLoader, device: str = None) -> float:
    """
    Evaluate the trained model on test data, with model type info.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    model.to(device)
    model.eval()
    
    # 判断模型类型
    if isinstance(model, PDFNet):
        model_type = "PDFNet"
    elif isinstance(model, XRDNetWithDynamic):
        model_type = "XRDNetWithDynamic"
    elif isinstance(model, XRDNet):
        model_type = "XRDNet"
    else:
        model_type = model.__class__.__name__
    
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f"Testing [{model_type}]")
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            test_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    test_accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"[{model_type}] Test Results: Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy
