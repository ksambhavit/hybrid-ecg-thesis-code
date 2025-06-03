# src/transformer_train_kfolds.py

import os, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from transformer_model_new import ECGTransformer  # your transformer that also takes metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ECGDataset(Dataset):
    def __init__(self, X, M, y, idxs):
        self.X = X[idxs]
        self.M = M[idxs]
        self.y = y[idxs]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]).float(),
            torch.from_numpy(self.M[i]).float(),
            torch.from_numpy(self.y[i]).float(),
        )


def compute_metrics(y_true, y_pred):
    y_t, y_p = y_true.astype(bool), y_pred.astype(bool)
    tp = np.logical_and(y_t, y_p).sum()
    fp = np.logical_and(~y_t, y_p).sum()
    fn = np.logical_and(y_t, ~y_p).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": prec, "recall": rec, "f1_score": f1}


def train_fold(X, M, y, tr_idx, va_idx, fold_id, patience=5):
    """
    Trains one CV fold, returns best_val_f1.
    """
    # — DataLoaders —
    B, W = 32, 4
    tr_dl = DataLoader(
        ECGDataset(X, M, y, tr_idx),
        batch_size=B, shuffle=True,
        num_workers=W, pin_memory=True
    )
    va_dl = DataLoader(
        ECGDataset(X, M, y, va_idx),
        batch_size=B, shuffle=False,
        num_workers=W // 2, pin_memory=True
    )

    # — Model —
    nc = y.shape[1]
    model = ECGTransformer(
        input_dim=X.shape[2],
        meta_dim=M.shape[1],
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        num_classes=nc,
        downsample=4
    ).to(device)

    # — Loss with class‐imbalance —
    pos = y[tr_idx].sum(axis=0)
    neg = len(tr_idx) - pos
    pos_w = torch.tensor(neg / pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # — Optimizer + cosine‐warmup LR —
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2
    )
    total_steps = len(tr_dl) * 50
    warmup = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / max(1, warmup),
            0.5 * (1.0 + math.cos(math.pi * (step - warmup) /
                                   max(1, total_steps - warmup)))
        )
    )

    best_f1, no_imp = 0.0, 0

    for ep in range(1, 51):
        # — TRAIN —
        model.train()
        running_loss = 0.0
        for Xb, Mb, yb in tqdm(
            tr_dl, desc=f"Fold{fold_id} Ep{ep:02d} Train", leave=False
        ):
            Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb, Mb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * Xb.size(0)

        train_loss = running_loss / len(tr_dl.dataset)

        # — VALIDATE —
        model.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for Xb, Mb, yb in tqdm(
                va_dl, desc=f"Fold{fold_id} Ep{ep:02d} Val", leave=False
            ):
                logits = model(Xb.to(device), Mb.to(device)).cpu().numpy()
                all_p.append((logits > 0.5).astype(float))
                all_t.append(yb.numpy())

        all_t = np.vstack(all_t)
        all_p = np.vstack(all_p)
        m = compute_metrics(all_t, all_p)

        print(
            f"Fold{fold_id} Ep{ep:02d} | "
            f"Loss {train_loss:.4f} | "
            f"Val F1 {m['f1_score']:.4f}"
        )

        # — Early stopping & save best —
        if m["f1_score"] > best_f1 + 1e-4:
            best_f1, no_imp = m["f1_score"], 0
            torch.save(
                model.state_dict(),
                os.path.join("processed_data", f"transformer_fold{fold_id}_best.pth")
            )
            print(f"🎉 new best F1={best_f1:.4f}")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"⚠️ Fold{fold_id} early stop at epoch {ep}")
                break

    return best_f1


def main():
    print(f"Using device: {device}\n")

    # — load preprocessed arrays —
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "processed_data")
    X = np.load(os.path.join(data_dir, "X_train.npy"), mmap_mode="r")
    M = np.load(os.path.join(data_dir, "meta_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))

    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    fold_f1s = []

    for fold_id, (tr, va) in enumerate(
        tqdm(kf.split(X), total=8, desc="CV Folds"), start=1
    ):
        print(f"\n=== Starting fold {fold_id} ===")
        f1 = train_fold(X, M, y, tr, va, fold_id, patience=5)
        fold_f1s.append(f1)

    # — final summary —
    print("\n=== CV results ===")
    for i, f1 in enumerate(fold_f1s, start=1):
        print(f"Fold {i}: {f1:.4f}")
    print(f"\nMean F1 = {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}\n")


if __name__ == "__main__":
    main()
