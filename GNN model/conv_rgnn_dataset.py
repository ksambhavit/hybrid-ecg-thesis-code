"""
Fixed‑topology ECG graph (12 leads) for PTB‑XL
---------------------------------------------
• limb loop  (I ‑ II ‑ III ‑ aVF ‑ aVL ‑ aVR ‑ I)
• chest loop (V1‑V2‑…‑V6‑V1)
• bridging  : I↔V6  + all‑limb→V6  + I→all‑chest
Node order = ['I','II','III','aVF','aVL','aVR','V1'…'V6']  (12)
Edge index is pre‑computed once.
"""
import os, numpy as np, pandas as pd, torch, wfdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer

# ------------------------------------------------------------------  fixed graph
_limb  = ['I','II','III','aVF','aVL','aVR']
_chest = [f'V{i}' for i in range(1,7)]
_bridge= [('I','V6')]
LEADS  = _limb + _chest
L2I    = {ld:i for i,ld in enumerate(LEADS)}

def _loop_edges(loop): return [(loop[i], loop[(i+1)%len(loop)]) for i in range(len(loop))]
edges  = _loop_edges(_limb)+_loop_edges(_chest)+_bridge
edges += [(u,'V6') for u in _limb] + [('I',v) for v in _chest]   # extra shortcuts
e_idx  = torch.tensor([[L2I[u],L2I[v]] for u,v in edges]+
                      [[L2I[v],L2I[u]] for u,v in edges]).t().contiguous()
EDGE_IDX = e_idx   # export

# ------------------------------------------------------------------  dataset
class ConvRGNNDataset(Dataset):
    def __init__(self, wfdb_dir, db_csv, scp_csv,
                 seg_len=2000, sampling_rate=100):
        self.wfdb_dir, self.seg_len = wfdb_dir, seg_len

        df = pd.read_csv(db_csv, index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(eval)

        # diag aggregation
        agg = pd.read_csv(scp_csv, index_col=0)
        agg = agg[agg.diagnostic == 1]
        def lab(codes): return list({agg.loc[c].diagnostic_class
                                     for c in codes if c in agg.index})
        df['diag_super'] = df.scp_codes.apply(lab)

        # meta
        meta = df[['age','sex']].copy()
        meta.age = (meta.age-meta.age.mean())/(meta.age.std()+1e-8)
        meta.sex = meta.sex.map({0:0,1:1}).fillna(0)
        self.meta = meta.values.astype(np.float32)

        # labels
        self.mlb   = MultiLabelBinarizer(classes=['NORM','MI','STTC','CD','HYP'])
        self.label = self.mlb.fit_transform(df['diag_super']).astype(np.float32)

        # filenames
        self.ids  = df.index.to_list()
        self.fmap = (df.filename_lr if sampling_rate==100 else df.filename_hr).to_dict()

    # -------------------------------------------------------------
    def __len__(self): return len(self.ids)

    # -------------------------------------------------------------
    def __getitem__(self, idx):
        rid   = self.ids[idx]
        path  = os.path.join(self.wfdb_dir, self.fmap[rid])
        sig   = wfdb.rdrecord(path).p_signal.T            # [12,N]

        # trim / pad to seg_len
        N = sig.shape[1]
        if N < self.seg_len:
            reps = int(np.ceil(self.seg_len / N))
            sig  = np.tile(sig, reps)[:, :self.seg_len]
        else:
            st = np.random.randint(0, N-self.seg_len+1)
            sig = sig[:, st:st+self.seg_len]

        # safe z‑score per lead
        std = sig.std(1, keepdims=True); std[std<1e-6] = 1
        x   = (sig - sig.mean(1, keepdims=True)) / std
        x   = torch.tensor(x, dtype=torch.float)          # [12,T]

        g       = Data(x=x, edge_index=EDGE_IDX)
        g.meta  = torch.tensor(self.meta[idx])
        g.y     = torch.tensor(self.label[idx])
        return g
