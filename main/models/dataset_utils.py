"""
Dataset Utility Functions

Provides functions for splitting data, creating data loaders, and balanced sampling.
"""

import numpy as np
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .gesture_dataset import GestureDataset


def get_train_val_test_split(
    dataset: GestureDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: GestureDataset instance
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_state: Random seed for reproducibility
        stratify: If True, preserve class distribution in splits
    
    Returns:
        (train_set, val_set, test_set) as torch.utils.data.Subset objects
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)
    
    # Train/temp split
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_state,
        stratify=labels if stratify else None
    )
    
    # Val/test split (from temp)
    temp_labels = labels[temp_idx]
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=random_state,
        stratify=temp_labels if stratify else None
    )
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Val set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    
    return train_set, val_set, test_set


def get_balanced_sampler(dataset: GestureDataset) -> WeightedRandomSampler:
    """
    Create a weighted sampler that balances class distribution during training.
    
    Args:
        dataset: GestureDataset instance
    
    Returns:
        WeightedRandomSampler that samples uniformly from all classes
    """
    class_counts = np.array([dataset.label_counts[label] for label in dataset.VALID_LABELS])
    total_samples = len(dataset)
    
    # Compute weight for each sample (inverse of class frequency)
    weights = []
    for label_idx in dataset.labels:
        class_count = class_counts[label_idx]
        weight = total_samples / (len(dataset.VALID_LABELS) * class_count + 1e-8)
        weights.append(weight)
    
    weights = np.array(weights)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def get_data_loaders(
    dataset: GestureDataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
    balance_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train/val/test sets.
    
    Args:
        dataset: GestureDataset instance
        batch_size: Batch size for loaders
        train_ratio, val_ratio, test_ratio: Split ratios
        random_state: Random seed
        stratify: If True, preserve class distribution
        balance_train: If True, use weighted sampler for training to balance classes
        num_workers: Number of worker processes for data loading
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_set, val_set, test_set = get_train_val_test_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        stratify=stratify
    )
    
    # Create sampler for training set (if balancing is desired)
    train_sampler = None
    if balance_train:
        # Get labels for train set
        train_labels = np.array(dataset.labels)[train_set.indices]
        class_counts_train = np.bincount(train_labels, minlength=len(dataset.VALID_LABELS))
        total_train = len(train_labels)
        
        weights = []
        for label in train_labels:
            class_count = class_counts_train[label]
            weight = total_train / (len(dataset.VALID_LABELS) * class_count + 1e-8)
            weights.append(weight)
        
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def analyze_dataset(dataset: GestureDataset):
    """
    Print dataset statistics and warnings.
    
    Args:
        dataset: GestureDataset instance
    """
    counts = dataset.get_class_counts()
    total = sum(counts.values())
    
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {total}")
    print("Class distribution:")
    
    for label in dataset.VALID_LABELS:
        count = counts[label]
        pct = 100 * count / total if total > 0 else 0
        print(f"  {label:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Warnings
    min_samples = min(counts.values()) if counts else 0
    if min_samples < 20:
        print(f"\n⚠️  Warning: Minimum class has only {min_samples} samples (recommend ≥20)")
    
    if total < 100:
        print(f"\n⚠️  Warning: Total dataset is small ({total} samples). Consider collecting more data.")
    
    # Imbalance warning
    if max(counts.values()) / (min(counts.values()) + 1e-8) > 5:
        print(f"\n⚠️  Warning: Significant class imbalance detected. Consider balanced sampling.")


if __name__ == "__main__":
    # Example usage
    dataset = GestureDataset("data/gestures", sequence_length=60)
    analyze_dataset(dataset)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=16,
        balance_train=True
    )
    
    print(f"\nLoaders created successfully")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
