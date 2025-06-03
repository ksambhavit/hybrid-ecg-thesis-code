# src/new_hyb_try/train.py

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from dataset import ECGDataset, collate_fn
from models  import HybridECGModel

# ─── 1) cuDNN benchmark ───
torch.backends.cudnn.benchmark = True

# ─── 2) Hyperparameters ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data & Sampling
BATCH_SIZE  = 32
ACCUM_STEPS = 2       # effective batch = 64

# Epochs & Early Stopping
NUM_EPOCHS           = 60
EARLY_STOP_PATIENCE  = 12

# Optimizer & LR
LR           = 3e-4
WEIGHT_DECAY = 1e-5

# OneCycleLR schedule
ONECYCLE_DIV_FACTOR = 25
ONECYCLE_PCT_START  = 0.2

# MixUp + Focal + WBCE + Label Smoothing
MIXUP_ALPHA         = 0.4
USE_MIXUP           = True
FOCAL_GAMMA         = 2.0
FOCAL_ALPHA         = 0.25
LABEL_SMOOTHING_EPS = 0.01

# Effective Number reweighting (for per-class pos_weights)
BETA = 0.9999

# Snapshot Ensembling
SNAPSHOT_CYCLES = 3
T0              = NUM_EPOCHS // SNAPSHOT_CYCLES  # epochs per cycle
T_MULT          = 1

# Threshold search range
THRESHOLDS = np.arange(0.3, 0.951, 0.01)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR                = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BEST_MODEL_STATE_PATH   = os.path.join(BASE_DIR, "src", "new_hyb_try", "hyb_try_model.pth")
BEST_MODEL_PT_PATH      = os.path.join(BASE_DIR, "src", "new_hyb_try", "best_model.pt")

# ─── 3) Loss & Helper Functions ───

def compute_effective_num_weights(all_labels, beta=BETA):
    """
    all_labels: numpy array [N, n_classes] of 0/1
    Returns: torch.Tensor [n_classes] with pos_weights
    """
    # Count positives per class
    pos_counts = all_labels.sum(axis=0) + 1e-6            # [n_classes]
    eff_num    = (1.0 - np.power(beta, pos_counts)) / (1.0 - beta)
    w          = 1.0 / eff_num
    w_norm     = w / np.max(w)
    return torch.from_numpy(w_norm.astype(np.float32))

def label_smoothing(y, eps=LABEL_SMOOTHING_EPS):
    # y: [B, n_classes], 0/1 → smoothed
    return y * (1 - eps) + (1 - y) * eps

def focal_loss(inputs, targets, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Sigmoid focal loss for multi-label.
    inputs: logits [B, n_classes]
    targets: [B, n_classes]
    """
    prob = torch.sigmoid(inputs)
    ce   = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # [B, n_classes]
    p_t  = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss    = alpha_t * loss
    return loss.mean()

def wbce_loss(inputs, targets, pos_weights):
    """
    Weighted BCE with logits using pos_weights per class.
    inputs: logits [B, n_classes], targets: [B, n_classes]
    pos_weights: [n_classes] tensor
    """
    return F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        pos_weight=pos_weights.to(inputs.device),
        reduction="mean"
    )

def combined_loss(inputs, targets, pos_weights):
    """
    70% focal + 30% weighted BCE
    """
    fl = focal_loss(inputs, targets)
    wb = wbce_loss(inputs, targets, pos_weights)
    return 0.7 * fl + 0.3 * wb

def mixup_data(x, meta, y, alpha=MIXUP_ALPHA):
    """
    MixUp: returns mixed_x, mixed_meta, (y_a, y_b, lam)
    """
    if alpha <= 0:
        return x, meta, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx_perm   = torch.randperm(batch_size).to(DEVICE)
    x_m        = lam * x + (1 - lam) * x[idx_perm]
    meta_m     = lam * meta + (1 - lam) * meta[idx_perm]
    y_a, y_b   = y, y[idx_perm]
    return x_m, meta_m, (y_a, y_b, lam)

# ─── 4) Training & Validation Loops ───

def train_one_epoch(model, loader, optimizer, pos_weights, scheduler_onecycle, scaler):
    model.train()
    total_loss  = 0.0
    # We will compute train‐metrics only once at the end of the epoch,
    # to avoid doubling forward passes every batch.
    all_logits = []
    all_trues  = []

    optimizer.zero_grad()
    train_bar = tqdm(loader, desc="Train", leave=False)

    for i, (x, meta, y, _) in enumerate(train_bar):
        # Move to device
        x    = x.to(DEVICE, non_blocking=True)   # [B,12,2500]
        meta = meta.to(DEVICE, non_blocking=True) # [B,meta_dim]
        y    = y.to(DEVICE, non_blocking=True)    # [B,n_classes]

        if USE_MIXUP:
            x_m, meta_m, (y_a, y_b, lam) = mixup_data(x, meta, y, MIXUP_ALPHA)
            y_a_sm = label_smoothing(y_a)
            y_b_sm = label_smoothing(y_b)
        else:
            lam    = 1.0
            y_sm   = label_smoothing(y)
            x_m    = x
            meta_m = meta
            y_a_sm = y_sm
            y_b_sm = y_sm

        # Mixed-precision forward
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x_m, meta_m)  # [B, n_classes]
            if USE_MIXUP:
                loss = lam * combined_loss(logits, y_a_sm, pos_weights) \
                     + (1 - lam) * combined_loss(logits, y_b_sm, pos_weights)
            else:
                loss = combined_loss(logits, y_a_sm, pos_weights)

            # scale by ACCUM_STEPS
            loss = loss / ACCUM_STEPS

        # Backward & step every ACCUM_STEPS
        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Step OneCycleLR once per effective batch
            scheduler_onecycle.step()

        total_loss += (loss.item() * ACCUM_STEPS) * x.size(0)

        # Collect logits & truths (on the unmixed x, to measure actual status)
        with torch.no_grad():
            logits_orig = model(x, meta)  # one extra forward per batch
            all_logits.append(torch.sigmoid(logits_orig).cpu().numpy())
            all_trues.append(y.cpu().numpy())

        train_bar.set_postfix(batch_loss=f"{(loss.item()*ACCUM_STEPS):.4f}")

    # If leftovers (when len(loader) * BATCH_SIZE not divisible by ACCUM_STEPS)
    if (len(loader) % ACCUM_STEPS) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Compute epoch train‐metrics ONCE
    epoch_loss = total_loss / len(loader.dataset)
    preds_arr  = np.vstack(all_logits)  # [N_train, n_classes]
    trues_arr  = np.vstack(all_trues)
    # Use threshold=0.5 here—will be very low at first, but will improve as logits spread
    pred_labels = (preds_arr > 0.5).astype(int)
    prec = precision_score(trues_arr, pred_labels, average="samples", zero_division=0)
    rec  = recall_score(trues_arr, pred_labels, average="samples", zero_division=0)
    f1   = f1_score(trues_arr, pred_labels, average="samples", zero_division=0)

    return epoch_loss, prec, rec, f1

def validate_one_epoch(model, loader, pos_weights):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_trues  = []

    val_bar = tqdm(loader, desc="Val  ", leave=False)
    with torch.no_grad():
        for x, meta, y, _ in val_bar:
            x    = x.to(DEVICE, non_blocking=True)
            meta = meta.to(DEVICE, non_blocking=True)
            y    = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x, meta)  # [B, n_classes]
                loss   = combined_loss(logits, label_smoothing(y), pos_weights)

            total_loss += loss.item() * x.size(0)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_trues.append(y.cpu().numpy())

    val_loss   = total_loss / len(loader.dataset)
    logits_arr = np.vstack(all_logits)
    trues_arr  = np.vstack(all_trues)
    return val_loss, logits_arr, trues_arr

def threshold_search(logits, trues, thresholds=THRESHOLDS):
    n_classes = trues.shape[1]
    best_thresholds = np.zeros(n_classes)

    # 1) Global threshold search
    best_f1_g, best_t_g = 0.0, 0.5
    for t in thresholds:
        preds = (logits >= t).astype(int)
        f1_   = f1_score(trues, preds, average="samples", zero_division=0)
        if f1_ > best_f1_g:
            best_f1_g, best_t_g = f1_, t

    # 2) Per-class refinement
    for c in range(n_classes):
        best_f1_c, best_t_c = 0.0, best_t_g
        for t in thresholds:
            preds_c = (logits[:, c] >= t).astype(int)
            f1_c    = f1_score(trues[:, c], preds_c, zero_division=0)
            if f1_c > best_f1_c:
                best_f1_c, best_t_c = f1_c, t
        best_thresholds[c] = best_t_c

    # Final ensemble‐style sample-wise F1
    final_preds = (logits >= best_thresholds).astype(int)
    best_f1     = f1_score(trues, final_preds, average="samples", zero_division=0)
    return best_thresholds, best_f1

# ─── 5) Main Training Loop ───
def main():
    # 5.1) Load dataset
    ds_full = ECGDataset(
        base_dir=BASE_DIR,
        split="train",
        augment=True
    )
    N       = len(ds_full)
    n_val   = int(0.2 * N)
    n_train = N - n_val

    train_ds, val_ds = random_split(
        ds_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    val_ds.dataset.augment = False

    # 5.2) Compute per-class pos_weights (effective number)
    all_labels_np = ds_full.Y  # [N, n_classes]
    pos_weights   = compute_effective_num_weights(all_labels_np, beta=BETA).to(DEVICE)

    # 5.3) Build WeightedRandomSampler properly
    train_indices = train_ds.indices            # subset of [0..N-1]
    class_w       = pos_weights.cpu().numpy()    # [n_classes]

    sample_weights = []
    for i in train_indices:
        y_i = all_labels_np[i]                   # [n_classes], 0/1
        # Sum up the class_weights for the labels that are present
        w   = class_w[y_i == 1].sum()
        if w <= 0.0:
            w = 1e-6  # if truly no positives, give tiny weight
        sample_weights.append(w)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,    # change to >0 if your CPU can handle I/O
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 5.4) Instantiate model
    seq_len_down = ds_full.L // 2    # 2500
    meta_dim     = ds_full.meta_dim
    n_classes    = ds_full.n_classes

    model = HybridECGModel(
        seq_len    = seq_len_down,
        meta_dim   = meta_dim,
        n_classes  = n_classes,
        d_cnn      = 64,
        msw_dim    = 64,
        msw_heads  = 4,
        graph_dim  = 64,
        graph_heads= 4,
        mlp_hidden = 128,
    ).to(DEVICE)

    # 5.5) Optimizer, OneCycleLR, CosineRestarts, AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * NUM_EPOCHS

    scheduler_onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = LR,
        total_steps    = total_steps,
        pct_start      = ONECYCLE_PCT_START,
        div_factor     = ONECYCLE_DIV_FACTOR,
        final_div_factor = 100.0,
        anneal_strategy = "cos",
        three_phase     = False
    )

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T0, T_mult=T_MULT
    )

    scaler = torch.amp.GradScaler()

    best_val_f1       = 0.0
    epochs_no_improve = 0
    snapshot_count    = 0
    saved_checkpoints = []

    # 5.6) Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_prec, tr_rec, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, pos_weights, scheduler_onecycle, scaler
        )

        # Print train metrics
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"TrainLoss={tr_loss:.4f} | "
            f"TrainPrec={tr_prec:.4f} | "
            f"TrainRec={tr_rec:.4f} | "
            f"TrainF1={tr_f1:.4f}"
        )

        # Step CosineAnnealingWarmRestarts once (per epoch)
        scheduler_cosine.step()

        # 5.7) Validation & threshold search
        val_loss_f, val_logits_f, val_trues_f = validate_one_epoch(model, val_loader, pos_weights)
        thresholds, val_f1_f = threshold_search(val_logits_f, val_trues_f)
        val_preds_f = (val_logits_f >= thresholds).astype(int)
        val_prec_f  = precision_score(val_trues_f, val_preds_f, average="samples", zero_division=0)
        val_rec_f   = recall_score(val_trues_f, val_preds_f, average="samples", zero_division=0)

        line = (
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"TrainLoss={tr_loss:.4f} | "
            f"ValLoss={val_loss_f:.4f} | "
            f"ValPrec={val_prec_f:.4f} | "
            f"ValRec={val_rec_f:.4f} | "
            f"ValF1={val_f1_f:.4f}"
        )

        # 5.8) Snapshot ensembling (save at each T0, up to SNAPSHOT_CYCLES)
        if (epoch % T0 == 0) and (snapshot_count < SNAPSHOT_CYCLES):
            snapshot_count += 1
            ckpt_path = os.path.join(
                BASE_DIR, "src", "new_hyb_try", f"snapshot_{snapshot_count}.pth"
            )
            torch.save({"model_state":model.state_dict(), "thresholds":thresholds}, ckpt_path)
            saved_checkpoints.append((val_f1_f, ckpt_path))
            print(f"→ Snapshot {snapshot_count} saved at epoch {epoch}.")

        # 5.9) Save best single model
        if val_f1_f > best_val_f1 + 1e-5:
            best_val_f1 = val_f1_f
            torch.save(
                {"model_state": model.state_dict(), "thresholds": thresholds},
                BEST_MODEL_STATE_PATH
            )
            torch.save(model.state_dict(), BEST_MODEL_PT_PATH)
            print(line + f"  → New best ValF1: {val_f1_f:.4f}. Saved.")
            epochs_no_improve = 0
        else:
            print(line)
            if val_f1_f < best_val_f1 - 1e-5:
                epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"No improvement in {EARLY_STOP_PATIENCE} epochs. Early stopping.\n")
            break

    # 5.10) Ensemble the top-3 snapshots
    if saved_checkpoints:
        saved_checkpoints.sort(key=lambda x: x[0], reverse=True)
        top_ckpts = saved_checkpoints[:SNAPSHOT_CYCLES]

        all_probs = []
        model.eval()
        with torch.no_grad():
            for _f1, path in top_ckpts:
                cp = torch.load(path, map_location=DEVICE)
                model.load_state_dict(cp["model_state"])
                model.eval()
                probs_list = []
                for x, meta, y, _ in tqdm(val_loader, desc="Ensemble Eval"):
                    x    = x.to(DEVICE, non_blocking=True)
                    meta = meta.to(DEVICE, non_blocking=True)
                    logits = model(x, meta)
                    probs_list.append(torch.sigmoid(logits).cpu().numpy())
                all_probs.append(np.vstack(probs_list))

        avg_probs      = np.mean(np.stack(all_probs, axis=0), axis=0)    # [N_val, n_classes]
        avg_thresholds = np.mean([torch.load(p)["thresholds"] for _, p in top_ckpts], axis=0)
        preds_ensemble = (avg_probs >= avg_thresholds).astype(int)
        val_truths     = val_trues_f  # last epoch’s truths
        ensemble_f1    = f1_score(val_truths, preds_ensemble, average="samples", zero_division=0)
        print(f"\n=== Ensemble F1 (top {SNAPSHOT_CYCLES}): {ensemble_f1:.4f} ===")

        np.save(
            os.path.join(BASE_DIR, "src", "new_hyb_try", "ensemble_thresholds.npy"),
            avg_thresholds
        )

    # 5.11) Final Full Validation with Best Model
    checkpoint = torch.load(
        BEST_MODEL_STATE_PATH,
        map_location=DEVICE,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    final_probs = []
    with torch.no_grad():
        for x, meta, y, _ in val_loader:
            x    = x.to(DEVICE, non_blocking=True)
            meta = meta.to(DEVICE, non_blocking=True)
            logits = model(x, meta)
            final_probs.append(torch.sigmoid(logits).cpu().numpy())

    final_probs      = np.vstack(final_probs)
    final_thresholds = checkpoint["thresholds"]
    final_preds      = (final_probs >= final_thresholds).astype(int)

    # Recompute final metrics
    _, _, val_trues_e = validate_one_epoch(model, val_loader, pos_weights)
    final_prec = precision_score(val_trues_e, final_preds, average="samples", zero_division=0)
    final_rec  = recall_score(val_trues_e, final_preds, average="samples", zero_division=0)
    final_f1   = f1_score(val_trues_e, final_preds, average="samples", zero_division=0)

    print(
        f"\n=== Final Validation with Best Model ===\n"
        f"ValPrecision={final_prec:.4f} | ValRecall={final_rec:.4f} | ValF1={final_f1:.4f}"
    )
    print("\nTraining complete.")
    print(f"Best Validation F1 = {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
