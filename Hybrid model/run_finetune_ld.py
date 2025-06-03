# ─── src/new_hyb_try/run_finetune_ld.py  (FULL SCRIPT) ─────────────────────
import argparse, os, math, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from models            import HybridECGModel
from dataset_leaddrop  import ECGDatasetLeadDrop as ECGDataset
from dataset           import collate_fn
from load_ckpt         import safe_load

# ─── CLI ────────────────────────────────────────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser(
        description="Lead-drop fine-tune starting from an existing checkpoint"
    )
    ap.add_argument("--ckpt",  default="hyb_try_model_ld.pth",
                    help="checkpoint filename inside src/new_hyb_try/")
    ap.add_argument("--p_drop", type=float, default=0.08,
                    help="probability to zero each lead (default 0.08)")
    ap.add_argument("--epochs", type=int,   default=20,
                    help="fine-tune epochs (default 20)")
    ap.add_argument("--lr",     type=float, default=5e-6,
                    help="learning-rate (default 5e-6)")
    return ap.parse_args()

# ─── TRAIN / VAL LOOP HELPERS ───────────────────────────────────────────────
@torch.inference_mode()
def evaluate(model, loader, thresholds):
    model.eval()
    probs, trues = [], []
    for x, meta, y, _ in loader:
        x, meta = x.to(device), meta.to(device)
        logits  = model(x, meta)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(y.numpy())
    probs = np.vstack(probs); trues = np.vstack(trues)
    preds = (probs >= thresholds).astype(int)
    f1    = f1_score(trues, preds, average="samples", zero_division=0)
    loss  = F.binary_cross_entropy(
        torch.from_numpy(probs), torch.from_numpy(trues).float()
    ).item()
    return loss, f1, probs, trues

def threshold_search(logits, labels):
    thr = np.linspace(0.1, 0.9, 81)
    best = 0.5; best_f1 = 0.0
    for t in thr:
        f1 = f1_score(labels, (logits >= t).astype(int),
                      average="samples", zero_division=0)
        if f1 > best_f1:
            best_f1, best = f1, t
    return np.full(labels.shape[1], best, dtype=np.float32)

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    global device
    args      = get_args()
    PROJ_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_path = os.path.join(PROJ_DIR, "src", "new_hyb_try", args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ── DATA ──
    ds = ECGDataset(base_dir=PROJ_DIR, split="train",
                    augment=True, p_drop=args.p_drop)
    n_val   = int(0.2 * len(ds))
    n_train = len(ds) - n_val
    train_set, val_set = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.augment = False      # no aug at val

    tr_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                           num_workers=2, pin_memory=True,
                           collate_fn=collate_fn)
    va_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                           num_workers=2, pin_memory=True,
                           collate_fn=collate_fn)

    # ── MODEL ──  (must match training width=64 etc.)
    model = HybridECGModel(
        seq_len   = ds.L // 2,
        meta_dim  = ds.meta_dim,
        n_classes = ds.n_classes,
        d_cnn=64, msw_dim=64, graph_dim=64,
        mlp_hidden=128
    ).to(device)

    # load checkpoint (strict=False so minor shape diffs are OK)
    ckpt = safe_load(model, ckpt_path, device=device)          # prints stats
    thresholds = ckpt.get("thresholds", None)
    if thresholds is None:
        print("⚠  No thresholds in ckpt – will search after 1st val pass")
        thresholds = np.full(ds.n_classes, 0.5, dtype=np.float32)

    # ── OPTIM / SCHED ──
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0
    for ep in range(1, args.epochs + 1):
        # ---- train --------------------------------------------------------
        model.train(); tr_loss, tr_logits, tr_trues = 0.0, [], []
        pbar = tqdm(tr_loader, leave=False, desc=f"Ep{ep:02d}")
        for x, meta, y, _ in pbar:
            x, meta, y = x.to(device), meta.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                logit = model(x, meta)
                loss  = F.binary_cross_entropy_with_logits(
                    logit, y.float())

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            tr_loss += loss.item() * x.size(0)
            tr_logits.append(torch.sigmoid(logit).detach().cpu().numpy())
            tr_trues .append(y.cpu().numpy())
        tr_loss /= len(train_set)
        tr_f1    = f1_score(np.vstack(tr_trues),
                            (np.vstack(tr_logits) >= thresholds).astype(int),
                            average="samples", zero_division=0)

        # ---- validate -----------------------------------------------------
        va_loss, va_f1, va_logits, va_trues = evaluate(model, va_loader, thresholds)
        # first epoch → (re)search thresholds
        if ep == 1 and "thresholds" not in ckpt:
            thresholds = threshold_search(va_logits, va_trues)
            va_loss, va_f1, *_ = evaluate(model, va_loader, thresholds)

        tag = " *" if va_f1 > best_f1 else ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model_state":model.state_dict(),
                        "thresholds": thresholds},
                       ckpt_path.replace(".pth","_ld_best.pth"))
        print(f"Ep{ep:02d} TL={tr_loss:.4f} TF1={tr_f1:.3f} "
              f"VL={va_loss:.4f} VF1={va_f1:.3f}{tag}")

    print(f"\nFine-tune finished.  Best Val-F1 = {best_f1:.4f}")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
