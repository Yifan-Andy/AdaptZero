import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlockLite(nn.Module):
    def __init__(self, hops, input_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.seq_len = hops + 1
        self.hidden_dim = hidden_dim

        # single-layer encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # shared attention projection layer
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batched_data):
        encoded = self.node_encoder(batched_data)
        encoded = self.norm(encoded)

        node_tensor = encoded[:, 0:1, :]
        neighbor_tensor = encoded[:, 1:, :]

        q = self.W(node_tensor)
        k = self.W(neighbor_tensor)

        attn_score = torch.sum(q * k, dim=-1, keepdim=True)
        attn_weight = F.softmax(attn_score, dim=1)

        neighbor_tensor = neighbor_tensor * attn_weight

        return node_tensor, neighbor_tensor

class MLPAblation(nn.Module):
    def __init__(self, hops, input_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.seq_len = hops + 1
        self.hidden_dim = hidden_dim

        # single-layer encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batched_data):
        encoded = self.node_encoder(batched_data)
        encoded = self.norm(encoded)

        # only encode with MLP with no query-aware compuation
        node_tensor = encoded[:, 0:1, :]
        neighbor_tensor = encoded[:, 1:, :]

        return node_tensor, neighbor_tensor
