# File: src/better_spatial_gnn.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import MixHopConv, global_mean_pool
from conv_summary import LeadResNet
from conv_rgnn_dataset import EDGE_IDX


class BetterSpatialGNN(nn.Module):
    """ResNet‑64  ➜  MixHop(1,2,3) × 3  ➜  LayerNorm + FC."""
    def __init__(self, seg_len, meta_dim, out_ch,
                 node_dim=64, hid=128, hops=(1, 2, 3)):
        super().__init__()
        self.summary = LeadResNet(seg_len, node_dim)

        # MixHop layers
        self.mh1 = MixHopConv(node_dim,          hid, hops)
        self.mh2 = MixHopConv(hid * len(hops),   hid, hops)
        self.mh3 = MixHopConv(hid * len(hops),   hid, hops)

        # keep a copy of the fixed edge index as a buffer
        self.register_buffer("edge_index_fixed", EDGE_IDX)

        self.norm = nn.LayerNorm(hid * len(hops))
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Sequential(
            nn.Linear(hid * len(hops) + meta_dim, 128),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, out_ch)
        )

    # ------------------------------------------------------------------ #
    def forward(self, data):
        B = int(data.batch.max()) + 1

        # 1. temporal summary for each lead
        x = data.x.view(B * 12, 1, -1)          # [B*12,1,T]
        x = self.summary(x)                     # [B*12,64]

        # 2. ensure edge_index is on the same device as x
        edge = self.edge_index_fixed.to(x.device)

        # 3. three MixHop blocks (residual-style depth)
        h = F.relu(self.mh1(x, edge))
        h = F.relu(self.mh2(h, edge))
        h = F.relu(self.mh3(h, edge))

        # 4. graph‑level embedding + meta fusion
        g = global_mean_pool(h, data.batch)     # [B, feat]
        g = self.norm(g)
        g = self.drop(g)
        g = torch.cat([g, data.meta.to(g.device)], 1)

        return self.fc(g)
