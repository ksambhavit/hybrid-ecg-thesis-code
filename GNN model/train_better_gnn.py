# File: src/train_better_gnn.py
# ------------------------------------------------------------
import os, numpy as np, torch, multiprocessing
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

from conv_rgnn_dataset import ConvRGNNDataset
from better_spatial_gnn import BetterSpatialGNN
# ------------------------------------------------------------
def macro_f1(y_true, y_pred):
    eps = 1e-8; f1 = []
    for c in range(y_true.shape[1]):
        tp = ((y_true[:,c]==1)&(y_pred[:,c]==1)).sum()
        fp = ((y_true[:,c]==0)&(y_pred[:,c]==1)).sum()
        fn = ((y_true[:,c]==1)&(y_pred[:,c]==0)).sum()
        p  = tp/(tp+fp+eps); r  = tp/(tp+fn+eps)
        f1.append(2*p*r/(p+r+eps))
    return np.mean(f1)

def collate_fn(lst):
    b = Batch.from_data_list(lst)
    b.meta = torch.stack([d.meta for d in lst]).to(b.x.device)
    b.y    = torch.stack([d.y    for d in lst]).to(b.x.device)
    return b
# ------------------------------------------------------------
def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    WFDB = os.path.join(ROOT, 'Data')
    CSV  = os.path.join(ROOT, 'data')
    DB_CSV, SCP_CSV = f'{CSV}/ptbxl_database.csv', f'{CSV}/scp_statements.csv'
    CKPT_DIR = os.path.join(ROOT, 'checkpoints_better'); os.makedirs(CKPT_DIR, exist_ok=True)

    DEV      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEG_LEN  = 2000
    BATCH    = 64
    LR, WD   = 1e-3, 5e-4
    EPOCHS, PATIENCE = 100, 10
    USE_AMP  = False          # switch to True after stable training

    # ---------------- dataset & split -----------------------
    ds = ConvRGNNDataset(WFDB, DB_CSV, SCP_CSV, seg_len=SEG_LEN)
    tr_idx, va_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=42)

    tr_loader = DataLoader(
        dataset      = torch.utils.data.Subset(ds, tr_idx),
        batch_size   = BATCH,
        shuffle      = True,
        num_workers  = 4,
        collate_fn   = collate_fn
    )
    va_loader = DataLoader(
        dataset      = torch.utils.data.Subset(ds, va_idx),
        batch_size   = BATCH,
        shuffle      = False,
        num_workers  = 4,
        collate_fn   = collate_fn
    )

    meta_dim, out_ch = ds.meta.shape[1], ds.label.shape[1]
    model = BetterSpatialGNN(SEG_LEN, meta_dim, out_ch).to(DEV)

    # class‑balance weights
    pos_cnt = ds.label[tr_idx].sum(0); pos_cnt[pos_cnt == 0] = 1
    pos_w   = torch.tensor(((len(tr_idx)-pos_cnt)/pos_cnt), device=DEV).clamp(1., 20.)
    crit    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                                       factor=0.5, patience=2, min_lr=1e-5)
    scaler = GradScaler(enabled=USE_AMP)

    best_f1, no_imp = 0.0, 0
    CKPT = f'{CKPT_DIR}/better_gnn.pth'

    # ================= training loop ========================
    for ep in range(1, EPOCHS + 1):
        # -------- train ----------
        model.train(); running = 0.0
        for batch in tqdm(tr_loader, desc=f"Train {ep:02d}"):
            batch = batch.to(DEV); opt.zero_grad(set_to_none=True)
            with autocast(device_type=DEV.type, enabled=USE_AMP):
                logits = model(batch)
                loss   = crit(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            running += loss.item()
        tr_loss = running / len(tr_loader)

        # -------- validate -------
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for batch in tqdm(va_loader, desc=f" Val  {ep:02d}"):
                batch = batch.to(DEV)
                with autocast(device_type=DEV.type, enabled=USE_AMP):
                    logits = model(batch)
                ps.append((torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int))
                ys.append(batch.y.cpu().numpy())
        f1 = macro_f1(np.vstack(ys), np.vstack(ps)); sched.step(f1)

        print(f"Ep{ep:02d}  loss={tr_loss:.4f}  Val F1={f1:.4f}")
        if f1 > best_f1 + 1e-4:
            best_f1, no_imp = f1, 0
            torch.save(model.state_dict(), CKPT)
            print("   → new best saved")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print("   → early stop")
                break
    print("Best Val F1 =", best_f1)

# ------------------------------------------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
