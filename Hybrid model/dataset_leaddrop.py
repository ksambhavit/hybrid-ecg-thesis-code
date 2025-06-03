# File: src/new_hyb_try/dataset_leaddrop.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDatasetLeadDrop(Dataset):
    """
    Exactly like your original ECGDataset, except that __getitem__ applies
    random “lead‐dropout” (zeroing out some of the 12 leads) when augment=True.
    """

    def __init__(self, base_dir: str, split: str, augment: bool, p_drop: float = 0.08):
        super().__init__()
        assert split in ("train", "val", "test")
        self.base_dir = base_dir
        self.split    = split
        self.augment  = augment
        self.p_drop   = p_drop

        # load the pre‐downsampled .npy files (these must already exist):
        #   X_{split}.npy  → shape (N, 5000, 12)
        #   meta_{split}.npy → shape (N, meta_dim)
        #   y_{split}.npy    → shape (N, n_classes)
        data_dir = os.path.join(base_dir, "processed_data")
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy"))      # (N, 5000, 12)
        self.M = np.load(os.path.join(data_dir, f"meta_{split}.npy"))   # (N, meta_dim)
        self.Y = np.load(os.path.join(data_dir, f"y_{split}.npy"))      # (N, n_classes)

        self.N         = self.X.shape[0]
        self.meta_dim  = self.M.shape[1]
        self.n_classes = self.Y.shape[1]
        self.L         = self.X.shape[1] // 2  # will downsample from 5000 → 2500

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1) grab one sample (shape (5000, 12) as a NumPy array)
        x_np = self.X[idx]    # shape: (5000, 12)
        m_np = self.M[idx]    # shape: (meta_dim,)
        y_np = self.Y[idx]    # shape: (n_classes,)

        # 2) downsample time axis by 2 → shape (2500, 12)
        x_down = x_np[::2, :]   # (2500, 12)

        # 3) transpose to (12, 2500), convert to torch.FloatTensor
        #    This is *absolutely critical* so that SATSE sees 12 channels.
        x = torch.from_numpy(x_down.astype(np.float32)).transpose(0, 1)  # → [12, 2500]

        # 4) meta and labels
        m = torch.from_numpy(m_np.astype(np.float32))   # [meta_dim]
        y = torch.from_numpy(y_np.astype(np.float32))   # [n_classes]

        # 5) if we're in augment mode, randomly drop some leads
        if self.augment:
            x = self._lead_dropout(x)

        return x, m, y, idx

    def _lead_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero out each of the 12 leads (rows) with probability self.p_drop.
        Input x: [12, L]
        Output x: [12, L] with some rows entirely set to zero.
        """
        x = x.clone()  # <-- do not use x.copy(), torch.Tensor has no .copy()

        for lead in range(12):
            if torch.rand(1).item() < self.p_drop:
                x[lead, :] = 0.0

        return x
