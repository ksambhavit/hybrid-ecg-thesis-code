import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1251):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


class ECGTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        meta_dim : int = 0,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 5,
        downsample: int = 4,
    ):
        super().__init__()
        # 1) strided conv embed
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3, stride=downsample)
        # 2) positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=(5000//downsample)+1)
        # 3) transformer encoder
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                         nhead=num_heads,
                                         dropout=dropout,
                                         batch_first=False)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        # 4) metadata MLP
        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_fc = nn.Linear(meta_dim, hidden_dim)
        # 5) pooling + head
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.norm   = nn.LayerNorm(hidden_dim)
        self.drop   = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, meta=None):
        # x: (B, T, C)
        x = x.permute(0,2,1)                  # → (B, C, T)
        x = self.conv(x)                     # → (B, D, T//ds)
        x = x.transpose(1,2)                 # → (B, T', D)
        x = self.pos_enc(x)
        x = x.permute(1,0,2)                 # → (T', B, D)
        x = self.transformer(x)              # → (T', B, D)
        x = x.permute(1,2,0)                 # → (B, D, T')
        x = self.pool(x).squeeze(-1)         # → (B, D)

        # incorporate metadata
        if self.meta_dim > 0 and meta is not None:
            m = self.meta_fc(meta)           # → (B, D)
            x = x + m

        x = self.norm(x)
        x = self.drop(x)
        return self.classifier(x)            # logits
