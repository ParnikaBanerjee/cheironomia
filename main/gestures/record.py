import numpy as np
import os
import time

DATA_DIR = "data/gestures"

def save_sequence(sequence, label):
    os.makedirs(DATA_DIR, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.npy"
    path = os.path.join(DATA_DIR, filename)

    np.save(path, sequence)
    print(f"[SAVED] {path}")
