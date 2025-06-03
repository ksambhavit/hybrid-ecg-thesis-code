# src/utils.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ECGLazyDataset(Dataset):
    def __init__(self, x_path, y_path, meta_path):
        # memmap so we never load the whole thing at once
        self.X    = np.load(x_path,   mmap_mode="r")
        self.y    = np.load(y_path,   mmap_mode="r")
        self.meta = np.load(meta_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()      # single-sample float32
        y = torch.from_numpy(self.y[idx]).float()
        m = torch.from_numpy(self.meta[idx]).float()
        return x, y, m

def get_dataloader(x_path, y_path, meta_path,
                   batch_size=1, shuffle=False):
    ds = ECGLazyDataset(x_path, y_path, meta_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,         # one worker: no extra memmap copies
        pin_memory=False,      # don’t pin—we’re streaming from disk
        persistent_workers=False,
    )

def compute_metrics(y_true, y_pred):
    # your existing code...
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall":    recall_score(y_true, y_pred, average="macro"),
        "f1_score":  f1_score(y_true, y_pred, average="macro"),
    }
