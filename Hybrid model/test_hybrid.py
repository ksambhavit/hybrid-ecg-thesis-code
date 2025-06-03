# File: src/new_hyb_try/test_hybrid.py
# ──────────────────────────────────────────────────────────────────────────────
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# ─── your local imports ────────────────────────────────────────────────────────
from load_ckpt import safe_load
from models     import HybridECGModel

# ─── 1) Configuration & paths ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

X_TEST_NPY    = os.path.join(PROCESSED_DIR, "X_test.npy")    # [N, 5000, 12]
META_TEST_NPY = os.path.join(PROCESSED_DIR, "meta_test.npy") # [N, meta_dim]
Y_TEST_NPY    = os.path.join(PROCESSED_DIR, "y_test.npy")    # [N, n_classes]

DOWNSAMPLE_FACTOR = 2  # 5000 → 2500

# ─── 2) Helper functions ──────────────────────────────────────────────────────
def compute_samplewise_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    prec = precision_score(y_true, y_pred, average="samples", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="samples", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="samples", zero_division=0)
    return prec, rec, f1

def load_and_preprocess_test_arrays():
    """
    - Loads X_test.npy (shape [N,5000,12]), down‐samples to [N,2500,12], then transposes to [N,12,2500].
    - Loads meta_test.npy ([N,meta_dim]) and y_test.npy ([N,n_classes]).
    """
    X = np.load(X_TEST_NPY, mmap_mode="r")    # shape: [N, 5000, 12]
    M = np.load(META_TEST_NPY)                # shape: [N, meta_dim]
    Y = np.load(Y_TEST_NPY)                   # shape: [N, n_classes]

    N, T, C = X.shape
    assert C == 12 and T == 5000, f"Expected X_test.npy shape [N,5000,12], got {X.shape}"

    # downsample time axis by factor of 2 → [N,2500,12]
    X_down = X[:, ::DOWNSAMPLE_FACTOR, :]

    # transpose to [N,12,2500], cast to float32
    X_down = X_down.transpose(0, 2, 1).astype(np.float32)

    return X_down, M.astype(np.float32), Y.astype(np.float32)

# ─── 3) DataLoader builder ────────────────────────────────────────────────────
def get_test_loader(X: np.ndarray, M: np.ndarray, Y: np.ndarray, batch_size: int = 32):
    tx = torch.from_numpy(X)  # [N,12,2500]
    tm = torch.from_numpy(M)  # [N,meta_dim]
    ty = torch.from_numpy(Y)  # [N,n_classes]
    ds = TensorDataset(tx, tm, ty)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader

# ─── 4) Main evaluation ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test a saved HybridECGModel on unseen data.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to saved checkpoint (e.g. hyb_try_model_ld.pth)")
    parser.add_argument("--batch", type=int, default=32, help="batch size for inference")
    args = parser.parse_args()

    # 4.1) Load & preprocess test arrays
    X_down, M_test, Y_test = load_and_preprocess_test_arrays()
    N, C, L = X_down.shape
    _, meta_dim   = M_test.shape
    _, n_classes  = Y_test.shape

    print(f"Using device: {DEVICE}")
    print(f"Loaded test set → {N} samples  |  seq_len={L}  |  meta_dim={meta_dim}  |  n_classes={n_classes}")

    # 4.2) Instantiate model with exactly the same architecture as training
    model = HybridECGModel(
        seq_len     = L,
        meta_dim    = meta_dim,
        n_classes   = n_classes,
        d_cnn       = 64,
        msw_dim     = 64,
        msw_heads   = 4,
        msw_window1 = 32,
        msw_window2 = 32,
        graph_layers= 2,
        graph_dim   = 64,
        graph_heads = 4,
        mlp_hidden  = 128,
    ).to(DEVICE)

    # 4.3) Load checkpoint (weights + per‐class thresholds if present)
    ckpt = safe_load(model, args.ckpt, DEVICE)
    model.eval()

    if "th" in ckpt:
        best_thresholds = np.array(ckpt["th"], dtype=np.float32)  # shape [n_classes]
        print(f"→ Using stored per‐class thresholds = {best_thresholds}")
    else:
        best_thresholds = np.full((n_classes,), 0.30, dtype=np.float32)
        print(f"⚠ No thresholds found → falling back to uniform 0.30")

    # 4.4) Inference
    test_loader = get_test_loader(X_down, M_test, Y_test, batch_size=args.batch)

    all_preds = []
    all_true  = []

    with torch.no_grad():
        for xb, mb, yb in tqdm(test_loader, desc="Test", leave=False):
            xb = xb.to(DEVICE)   # [B,12,2500]
            mb = mb.to(DEVICE)   # [B,meta_dim]
            logits = model(xb, mb)              # [B,n_classes]
            probs  = torch.sigmoid(logits).cpu().numpy()  # [B,n_classes]
            preds  = (probs >= best_thresholds).astype(int)
            all_preds.append(preds)
            all_true.append(yb.cpu().numpy())

    all_preds = np.vstack(all_preds)  # [N,n_classes]
    all_true  = np.vstack(all_true)   # [N,n_classes]

    prec, rec, f1 = compute_samplewise_metrics(all_true, all_preds)
    print("\n=== Final Test Metrics ===")
    print(f"Precision       = {prec:0.4f}")
    print(f"Recall          = {rec:0.4f}")
    print(f"Sample‐wise F1  = {f1:0.4f}")

if __name__ == "__main__":
    main()
