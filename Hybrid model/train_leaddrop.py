# src/new_hyb_try/train_leaddrop.py
"""
Train from scratch **with lead–dropout**.
Everything is copied from your latest `train.py` except:
  • imports ECGDatasetLeadDrop
  • default p_drop=0.2
  • prints "LD" for clarity
Nothing else changed – so the model architecture, loss functions,
LR schedule, snapshot logic … all remain identical.
"""

import os, numpy as np, torch, argparse
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# ─── your local modules ────────────────────────────────────────────────
from dataset_leaddrop import ECGDatasetLeadDrop as ECGDataset
from models   import HybridECGModel
from train    import (compute_effective_num_weights, mixup_data,
                      label_smoothing, focal_loss, wbce_loss, combined_loss,
                      threshold_search, validate_one_epoch, THRESHOLDS)

from train import (DEVICE, BATCH_SIZE, ACCUM_STEPS, NUM_EPOCHS,
                   EARLY_STOP_PATIENCE, LR, WEIGHT_DECAY,
                   ONECYCLE_DIV_FACTOR, ONECYCLE_PCT_START,
                   SNAPSHOT_CYCLES, T0, T_MULT, BETA,
                   BASE_DIR, BEST_MODEL_STATE_PATH, BEST_MODEL_PT_PATH,
                   RANDOM_SEED, mixup_data, THRESHOLDS)

torch.backends.cudnn.benchmark = True

# ─── Lead-drop argument ------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--p_drop", type=float, default=0.20,
                    help="probability of *each* lead being zeroed (train only)")
args = parser.parse_args()
P_DROP = args.p_drop

print(f"\n⚡ Lead-drop training: p_drop = {P_DROP:.2f}\n")

# --------------------------------------------------------------------- #
def collate_fn(batch):
    # identical to original collate_fn in dataset.py
    xs, metas, ys, idxs = zip(*batch)
    xs    = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    metas = torch.tensor(np.stack(metas, axis=0), dtype=torch.float32)
    ys    = torch.tensor(np.stack(ys, axis=0),    dtype=torch.float32)
    return xs, metas, ys, idxs

# --------------------------- TRAIN LOOP ------------------------------ #
def train_one_epoch(model, loader, optimizer, pos_w, sched_oc, scaler):
    model.train()
    total_loss, all_logits, all_trues = 0., [], []
    optimizer.zero_grad()
    bar = tqdm(loader, desc="Train-LD", leave=False)

    for i,(x,meta,y,_) in enumerate(bar):
        x, meta, y = x.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)

        x_m, meta_m, (y_a, y_b, lam) = mixup_data(x, meta, y, alpha=0.4)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x_m, meta_m)
            loss = lam*combined_loss(logits, label_smoothing(y_a), pos_w) + \
                   (1-lam)*combined_loss(logits, label_smoothing(y_b), pos_w)
            loss = loss / ACCUM_STEPS
        scaler.scale(loss).backward()

        if (i+1)%ACCUM_STEPS==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            sched_oc.step()

        total_loss += loss.item()*ACCUM_STEPS * x.size(0)
        with torch.no_grad():
            all_logits.append(torch.sigmoid(model(x,meta)).cpu().numpy())
            all_trues.append(y.cpu().numpy())
        bar.set_postfix(b_loss=f"{loss.item()*ACCUM_STEPS:.3f}")

    # left-over gradient step
    if (len(loader)%ACCUM_STEPS)!=0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer); scaler.update()

    preds = (np.vstack(all_logits) > 0.5).astype(int)
    trues = np.vstack(all_trues)
    f1 = f1_score(trues, preds, average="samples", zero_division=0)
    prec = precision_score(trues, preds, average="samples", zero_division=0)
    rec  = recall_score(trues, preds, average="samples", zero_division=0)
    return total_loss/len(loader.dataset), prec, rec, f1

# -------------------------- MAIN ------------------------------------- #
def main():
    # 1) Dataset -------------------------------------------------------
    ds_full = ECGDataset(base_dir=BASE_DIR, split="train",
                         augment=True, p_drop=P_DROP)
    N = len(ds_full); n_val=int(0.2*N); n_tr=N-n_val
    train_ds, val_ds = random_split(ds_full,[n_tr,n_val],
                    generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_ds.dataset.augment=False      # turn off lead-drop for val

    # Weighted sampler identical to previous script -------------------
    pos_w = compute_effective_num_weights(ds_full.Y, beta=BETA).to(DEVICE)
    class_w = pos_w.cpu().numpy()
    sample_w = [class_w[ds_full.Y[i]==1].sum() for i in train_ds.indices]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    tr_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,sampler=sampler,
                           num_workers=0,pin_memory=True,collate_fn=collate_fn)
    val_loader= DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,
                           num_workers=0,pin_memory=True,collate_fn=collate_fn)

    # 2) Model / opt / sched ------------------------------------------
    seq_len=ds_full.L//2; meta_dim=ds_full.meta_dim; n_cls=ds_full.n_classes
    model = HybridECGModel(seq_len,meta_dim,n_cls,
                           d_cnn=64,msw_dim=64,msw_heads=4,
                           graph_dim=64,graph_heads=4,mlp_hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    steps = len(tr_loader)*NUM_EPOCHS
    sched_oc = torch.optim.lr_scheduler.OneCycleLR(
        opt,max_lr=LR,total_steps=steps,
        pct_start=ONECYCLE_PCT_START,div_factor=ONECYCLE_DIV_FACTOR,
        final_div_factor=100.0,three_phase=False,anneal_strategy="cos")
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,T_0=T0,T_mult=T_MULT)
    scaler = torch.amp.GradScaler()

    best_f1, no_imp = 0.0, 0
    for epoch in range(1,NUM_EPOCHS+1):
        tl,tp,tr,tf = train_one_epoch(model,tr_loader,opt,pos_w,sched_oc,scaler)
        vl,logits,trues = validate_one_epoch(model,val_loader,pos_w)
        th, vf1 = threshold_search(logits,trues)
        vp = precision_score(trues,(logits>=th).astype(int),
                             average="samples",zero_division=0)
        vr = recall_score(trues,(logits>=th).astype(int),
                          average="samples",zero_division=0)
        tag = "  → new best !" if vf1>best_f1+1e-5 else ""
        print(f"Epoch{epoch:02d} TL={tl:.4f} TF1={tf:.3f} | "
              f"VL={vl:.4f} VF1={vf1:.3f}{tag}")
        if vf1>best_f1+1e-5:
            best_f1, no_imp = vf1, 0
            torch.save({"model_state":model.state_dict(),"th":th},
                       "hyb_try_model_ld.pth")
        else:
            no_imp += 1
        sched_cos.step()
        if no_imp>=EARLY_STOP_PATIENCE:
            print("Early-stop\n"); break

    print(f"\nLead-drop training finished.  Best Val F1 = {best_f1:.4f}")

if __name__ == "__main__":
    main()
