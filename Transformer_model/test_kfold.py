# src/test_kfold.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_model_new import ECGTransformer    # <-- your k-fold model
from utils import compute_metrics                  # <-- same compute_metrics you used above

# ——— Configuration ———
N_FOLDS   = 8
BATCH     = 32
THRESHOLD = 0.5
CKPT_FMT  = "transformer_fold{fold}_best.pth"
OUT_PATH  = "kfold_test_results.npz"

# ——— Paths & Device ———
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "processed_data")
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader(X, M, y, batch_size=BATCH):
    # copy into a writable np array, then float32 tensors
    X = np.array(X);  M = np.array(M);  y = np.array(y)
    tx = torch.tensor(X, dtype=torch.float32)
    tm = torch.tensor(M, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(tx, tm, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)


def test_kfold():
    # 1) load test arrays
    X_test = np.load(os.path.join(data_dir, "X_test.npy"),   mmap_mode="r")
    M_test = np.load(os.path.join(data_dir, "meta_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    loader   = get_test_loader(X_test, M_test, y_test, batch_size=BATCH)
    n_samples, _, in_dim = X_test.shape
    meta_dim             = M_test.shape[1]
    num_classes          = y_test.shape[1]

    # 2) accumulate probs
    probs_sum = np.zeros((n_samples, num_classes), dtype=np.float32)

    for fold in range(1, N_FOLDS + 1):
        ckpt_file = os.path.join(data_dir, CKPT_FMT.format(fold=fold))
        print(f"\n⏳ Loading fold {fold} checkpoint: {ckpt_file}")

        # re‐instantiate model exactly as in training
        model = ECGTransformer(
            input_dim   = in_dim,
            meta_dim    = meta_dim,
            hidden_dim  = 128,
            num_layers  = 4,
            num_heads   = 8,
            dropout     = 0.1,
            num_classes = num_classes,
            downsample  = 4
        ).to(device)

        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # run inference and accumulate
        idx = 0
        with torch.no_grad():
            for Xb, Mb, _ in tqdm(loader, desc=f"Fold {fold} ▶ Test", leave=False):
                Xb = Xb.to(device); Mb = Mb.to(device)
                logits = model(Xb, Mb)
                probs  = torch.sigmoid(logits).cpu().numpy()
                batch  = probs.shape[0]
                probs_sum[idx:idx+batch] += probs
                idx += batch

    # 3) average + threshold
    probs_avg = probs_sum / N_FOLDS
    y_pred    = (probs_avg > THRESHOLD).astype(np.float32)

    # 4) compute final metrics
    metrics = compute_metrics(y_test, y_pred)
    print("\n=== K-Fold Ensemble Test ===")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")

    # 5) save
    out_file = os.path.join(data_dir, OUT_PATH)
    print(f"\n💾 Saving predictions to {out_file}")
    np.savez(
        out_file,
        y_true = y_test,
        probs  = probs_avg,
        y_pred = y_pred
    )


if __name__ == "__main__":
    test_kfold()
