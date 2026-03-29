"""
Gesture Dataset Loader for LSTM Classification

Loads normalized gesture sequences from data/gestures/ directory and provides
torch.data.Dataset interface for training.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    """
    PyTorch Dataset for gesture sequences.
    
    Loads .npy files from data/gestures/ directory. Each file is named:
        {label}_{timestamp}.npy
    
    where label is one of: start, stop, accent, cut, pattern_change
    
    Each .npy file contains a normalized sequence of shape (T, 6, 3):
        T = variable time steps
        6 = joints (left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist)
        3 = x, y, z coordinates
    
    This dataset:
    1. Pads/truncates sequences to fixed length (default: 60 frames)
    2. Flattens (T, 6, 3) → (T, 18)
    3. Labels mapped to integers
    4. Returns torch tensors
    """
    
    VALID_LABELS = ["start", "stop", "accent", "cut", "pattern_change"]
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(VALID_LABELS)}
    IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
    
    def __init__(
        self,
        data_dir: str = "data/gestures",
        sequence_length: int = 60,
        normalize_sequences: bool = False
    ):
        """
        Args:
            data_dir: Path to directory containing .npy files
            sequence_length: Fixed length to pad/truncate sequences to
            normalize_sequences: If True, normalize each sequence to [0, 1] range
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.normalize_sequences = normalize_sequences
        
        # Load dataset
        self.sequences = []
        self.labels = []
        self.label_counts = {label: 0 for label in self.VALID_LABELS}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all .npy files from data_dir."""
        if not self.data_dir.exists():
            print(f"Warning: {self.data_dir} does not exist. Creating empty dataset.")
            return
        
        npy_files = sorted(self.data_dir.glob("*.npy"))
        
        if not npy_files:
            print(f"Warning: No .npy files found in {self.data_dir}")
            return
        
        for file_path in npy_files:
            # Extract label from filename: {label}_{timestamp}.npy
            stem = file_path.stem
            parts = stem.split("_")
            
            if len(parts) < 2:
                print(f"Skipping {file_path.name}: invalid filename format")
                continue
            
            label = parts[0]
            
            if label not in self.VALID_LABELS:
                print(f"Skipping {file_path.name}: unknown label '{label}'")
                continue
            
            try:
                sequence = np.load(file_path)  # Shape: (T, 6, 3)
                
                if sequence.ndim != 3 or sequence.shape[1:] != (6, 3):
                    print(f"Skipping {file_path.name}: unexpected shape {sequence.shape}")
                    continue
                
                self.sequences.append(sequence)
                self.labels.append(self.LABEL_TO_IDX[label])
                self.label_counts[label] += 1
                
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
        
        print(f"\nDataset loaded: {len(self.sequences)} sequences")
        print("Label distribution:")
        for label, count in self.label_counts.items():
            print(f"  {label}: {count}")
    
    def _process_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Process a sequence:
        1. Pad/truncate to fixed length
        2. Flatten (T, 6, 3) → (T, 18)
        3. Optionally normalize to [0, 1]
        
        Args:
            sequence: Shape (T, 6, 3)
        
        Returns:
            Tensor of shape (sequence_length, 18)
        """
        T = sequence.shape[0]
        
        if T < self.sequence_length:
            # Pad with zeros at the end
            padded = np.zeros((self.sequence_length, 6, 3), dtype=np.float32)
            padded[:T] = sequence
            sequence = padded
        else:
            # Truncate from the beginning (preserve ending dynamics)
            sequence = sequence[-self.sequence_length:]
        
        # Flatten (T, 6, 3) → (T, 18)
        sequence = sequence.reshape(self.sequence_length, -1)  # (T, 18)
        
        # Optional normalization
        if self.normalize_sequences:
            min_val = sequence.min()
            max_val = sequence.max()
            if max_val - min_val > 1e-6:  # Avoid division by zero
                sequence = (sequence - min_val) / (max_val - min_val)
        
        return torch.from_numpy(sequence).float()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (sequence_tensor, label_idx)
            where sequence_tensor is shape (sequence_length, 18)
            and label_idx is int in [0, 4]
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        sequence_tensor = self._process_sequence(sequence)
        
        return sequence_tensor, label
    
    def get_label_name(self, idx: int) -> str:
        """Convert label index to label name."""
        return self.IDX_TO_LABEL.get(idx, "unknown")
    
    def get_class_counts(self) -> Dict[str, int]:
        """Return count of samples per class."""
        return self.label_counts.copy()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        Returns weights inversely proportional to class frequency.
        """
        counts = np.array([self.label_counts[label] for label in self.VALID_LABELS])
        total = counts.sum()
        weights = total / (len(self.VALID_LABELS) * counts + 1e-8)
        return torch.from_numpy(weights).float()
