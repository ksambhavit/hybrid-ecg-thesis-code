# model.py

import torch
import torch.nn as nn
import math

class ECGGNNTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 1000,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 1,
        num_graph_layers: int = 1,
        num_classes: int = 5,
    ):
        super(ECGGNNTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. Per-lead Transformer Encoder
        self.input_proj = nn.Linear(1, d_model)
        self.positional_encoding = self._make_positional_encoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # 2. Inter-lead Graph Attention
        graph_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.lead_graph_encoder = nn.TransformerEncoder(
            graph_layer,
            num_layers=num_graph_layers
        )

        # 3. Final classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def _make_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x: (batch, seq_len=1000, num_leads=12)
        batch_size, seq_len, num_leads = x.shape
        if seq_len == 12 and num_leads == 1000:
            x = x.transpose(1, 2)
            batch_size, seq_len, num_leads = x.shape

        # 1) Per-lead encoding
        lead_feats = []
        pe = self.positional_encoding[:, :seq_len, :].to(x.device)  # (1, seq_len, d_model)
        for lead_idx in range(num_leads):
            lead_seq = x[:, :, lead_idx].unsqueeze(-1)    # (batch, seq_len, 1)
            emb = self.input_proj(lead_seq)               # (batch, seq_len, d_model)
            emb = emb + pe                                # add positional
            out = self.transformer_encoder(emb)           # (batch, seq_len, d_model)
            feat = out.mean(dim=1)                        # (batch, d_model)
            lead_feats.append(feat)

        lead_feats = torch.stack(lead_feats, dim=1)       # (batch, 12, d_model)

        # 2) Inter-lead graph attention
        lead_out = self.lead_graph_encoder(lead_feats)    # (batch, 12, d_model)
        global_feat = lead_out.mean(dim=1)                # (batch, d_model)

        # 3) Classification
        logits = self.classifier(global_feat)             # (batch, num_classes)
        return logits
