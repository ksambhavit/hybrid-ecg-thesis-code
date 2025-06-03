# src/new_hyb_try/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.signal as ss

class ECGDataset(Dataset):
    def __init__(self, base_dir, split="train", augment=False, sr=500):
        super().__init__()
        data_dir = os.path.join(base_dir, "processed_data")
        x_path   = os.path.join(data_dir, f"X_{split}.npy")
        meta_path= os.path.join(data_dir, f"meta_{split}.npy")
        y_path   = os.path.join(data_dir, f"y_{split}.npy")

        self.X = np.load(x_path, mmap_mode="r")        # [N,12,L]
        self.M = np.load(meta_path)                    # [N,meta_dim]
        self.Y = np.load(y_path)                       # [N,n_classes]

        self.N, self.C, self.L = self.X.shape
        self.meta_dim   = self.M.shape[1]
        self.n_classes  = self.Y.shape[1]

        self.augment = augment
        self.sr      = sr

    def __len__(self):
        return self.N

    def _time_warp(self, x):
        warped = np.zeros_like(x)
        for i in range(self.C):
            lead = x[i]
            L0   = lead.shape[-1]
            factor  = random.uniform(0.8, 1.2)
            new_len = int(L0 * factor)
            orig_idx= np.arange(L0)
            new_idx = np.linspace(0, L0-1, new_len)
            warped_lead = np.interp(new_idx, orig_idx, lead)
            if warped_lead.shape[-1] > L0:
                start = (warped_lead.shape[-1] - L0)//2
                warped[i] = warped_lead[start:start+L0]
            else:
                pad_l = (L0 - warped_lead.shape[-1])//2
                pad_r = L0 - warped_lead.shape[-1] - pad_l
                warped[i] = np.pad(warped_lead, (pad_l, pad_r), mode="constant")
        return warped

    def _add_noise(self, x, sigma=0.01):
        ranges = (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))
        noise  = np.random.randn(*x.shape) * sigma * ranges
        return x + noise

    def __getitem__(self, idx):
        x     = self.X[idx].astype(np.float32)       # [12, L]
        meta  = torch.from_numpy(self.M[idx]).float()  # [meta_dim]
        label = torch.from_numpy(self.Y[idx]).float()  # [n_classes]

        if self.augment:
            # (1) Random per-lead gain in [0.9,1.1]
            gains = np.random.uniform(0.9, 1.1, size=(self.C,1))
            x     = x * gains

            # (2) 50 Hz notch‐filter (IIR notch Q=30)
            b, a = ss.iirnotch(50.0, 30.0, fs=self.sr)
            for i in range(self.C):
                x[i] = ss.filtfilt(b, a, x[i])

            # (3) 50 % chance time‐warp
            if random.random() < 0.5:
                x = self._time_warp(x)
            # (4) 50 % chance additive noise
            if random.random() < 0.5:
                x = self._add_noise(x, sigma=0.01)

        x_torch = torch.from_numpy(x).float()  # [12, L]
        return x_torch, meta, label, idx

def collate_fn(batch):
    x_list, meta_list, y_list, idx_list = zip(*batch)
    x_batch    = torch.stack(x_list)
    meta_batch = torch.stack(meta_list)
    y_batch    = torch.stack(y_list)
    return x_batch, meta_batch, y_batch, list(idx_list)
