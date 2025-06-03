# File: src/test_better_gnn.py
# ------------------------------------------------------------
import os, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from conv_rgnn_dataset import ConvRGNNDataset
from better_spatial_gnn  import BetterSpatialGNN
# ------------------------------------------------------------
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WFDB    = os.path.join(ROOT, 'Data')
CSV_DIR = os.path.join(ROOT, 'data')
DB_CSV  = f'{CSV_DIR}/ptbxl_database.csv'
SCP_CSV = f'{CSV_DIR}/scp_statements.csv'
CKPT    = os.path.join(ROOT, 'checkpoints_better', 'better_gnn.pth')

SEG_LEN = 2000
BATCH   = 64
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------------------------
def macro_f1(y_true, y_pred):
    eps, C = 1e-8, y_true.shape[1]; f1=[]
    for c in range(C):
        tp=((y_true[:,c]==1)&(y_pred[:,c]==1)).sum()
        fp=((y_true[:,c]==0)&(y_pred[:,c]==1)).sum()
        fn=((y_true[:,c]==1)&(y_pred[:,c]==0)).sum()
        p=tp/(tp+fp+eps); r=tp/(tp+fn+eps)
        f1.append(2*p*r/(p+r+eps))
    return np.mean(f1), np.mean(p), np.mean(r)

def collate_fn(lst):
    b = Batch.from_data_list(lst)
    b.meta = torch.stack([d.meta for d in lst]).to(b.x.device)
    b.y    = torch.stack([d.y    for d in lst]).to(b.x.device)
    return b
# ------------------------------------------------------------
def main():
    # ---------- load dataset (all 10 folds) ------------------
    ds = ConvRGNNDataset(WFDB, DB_CSV, SCP_CSV, seg_len=SEG_LEN)

    # ---------- row positions belonging to strat_fold == 10 --
    meta = pd.read_csv(DB_CSV)
    test_idx = np.where(meta.strat_fold.values == 10)[0]      # <─ key line

    test_loader = DataLoader(
        Subset(ds, test_idx.tolist()),
        batch_size  = BATCH,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = collate_fn
    )

    # ---------- model ----------------------------------------
    meta_dim, out_ch = ds.meta.shape[1], ds.label.shape[1]
    model = BetterSpatialGNN(SEG_LEN, meta_dim, out_ch).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    # ---------- inference ------------------------------------
    ys, ps = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            batch  = batch.to(DEVICE)
            logits = model(batch)
            preds  = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            ys.append(batch.y.cpu().numpy())
            ps.append(preds)

    y_true, y_pred = np.vstack(ys), np.vstack(ps)
    f1, prec, rec  = macro_f1(y_true, y_pred)
    print(f"\n★ Test macro‑F1 = {f1:0.4f}  (P={prec:0.3f}, R={rec:0.3f})")

# ------------------------------------------------------------
if __name__ == '__main__':
    main()
