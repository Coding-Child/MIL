import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FeedForward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, query=None, src_mask=None, src_key_padding_mask=None):
        if query is not None:
            attn_output, _ = self.multihead_attn(query, src, src, attn_mask=src_mask,
                                                 key_padding_mask=src_key_padding_mask)
        else:
            attn_output, _ = self.multihead_attn(src, src, src, attn_mask=src_mask,
                                                 key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Sublayer 2: Feedforward network
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model,
                                                             num_heads,
                                                             dim_feedforward=d_ff,
                                                             dropout=dropout) for _ in range(num_layers)])

    def forward(self, query, features, mask):
        out = features

        for layer in self.layers:
            out = layer(out, query=query, src_key_padding_mask=mask)

        return out
