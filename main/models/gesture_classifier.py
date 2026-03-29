"""
LSTM-based Gesture Classifier

Architecture: 2-layer LSTM with dropout, followed by fully connected layer
Input: (batch, 60, 18) sequences
Output: (batch, 5) logits for 5 gesture classes
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple


class GestureClassifierLSTM(nn.Module):
    """
    LSTM-based classifier for gesture recognition.
    
    Architecture:
    - Input embedding (optional)
    - 2-layer LSTM with hidden_size=64
    - Dropout (p=0.3)
    - Fully connected layer → softmax over 5 classes
    
    Input shape: (batch_size, sequence_length, input_size)
                 (batch_size, 60, 18)
    
    Output shape: (batch_size, num_classes)
                 (batch_size, 5)
    """
    
    def __init__(
        self,
        input_size: int = 18,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Args:
            input_size: Number of input features (18 = 6 joints * 3 coordinates)
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (5 gestures)
            dropout: Dropout probability
            bidirectional: If True, use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 60, 18)
        
        Returns:
            Tensor of shape (batch_size, num_classes)
            Example: (32, 5)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        
        # Take the last hidden state
        if self.bidirectional:
            # Concatenate last states from both directions
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, hidden_size * 2)
        else:
            h_last = h_n[-1]  # (batch_size, hidden_size)
        
        # Apply dropout
        h_last = self.dropout(h_last)
        
        # Fully connected layer
        logits = self.fc(h_last)  # (batch_size, num_classes)
        
        return logits
    
    def save(self, path: str):
        """Save model state dict to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model state dict from file."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
    
    def summary(self):
        """Print model architecture summary."""
        print("\n=== Model Architecture ===")
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Num layers: {self.num_layers}")
        print(f"Bidirectional: {self.bidirectional}")
        print(f"Num classes: {self.num_classes}")
        print("\nModel layers:")
        for name, module in self.named_modules():
            if name:
                print(f"  {name}: {module}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


def create_model(
    input_size: int = 18,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 5,
    dropout: float = 0.3,
    bidirectional: bool = False,
    device: str = "cpu"
) -> GestureClassifierLSTM:
    """
    Factory function to create and initialize a gesture classifier.
    
    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout: Dropout probability
        bidirectional: If True, use bidirectional LSTM
        device: Device to move model to ("cpu" or "cuda")
    
    Returns:
        GestureClassifierLSTM instance on specified device
    """
    model = GestureClassifierLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = create_model(device=device)
    model.summary()
    
    # Test forward pass
    batch_size = 8
    seq_length = 60
    input_size = 18
    
    x = torch.randn(batch_size, seq_length, input_size).to(device)
    output = model(x)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, 5)")
    
    # Test save/load
    test_path = "models/test_model.pth"
    model.save(test_path)
    
    model2 = create_model(device=device)
    model2.load(test_path)
    print(f"\n✓ Model save/load test passed")
    
    # Cleanup
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
