import torch
import math
import torch.nn as nn
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        # define activate function
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Compute the positional encodings in advance
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StepEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.step_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.step_embedding(t)
        return emb


class TransformerLayer(nn.Module):

    def __init__(self, T, max_seq_length, d_model=128, n_head=4, num_encoder_layers=4):
        super(TransformerLayer, self).__init__()

        # define the step embedding layer
        self.step_encoder = StepEmbedding(T, 2 * d_model, 2 * d_model)

        # define the transformer layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(2 * d_model, n_head, batch_first=True, dropout=0.3),
            num_encoder_layers)
        self.positional_encoding = PositionalEncoding(2 * d_model, max_seq_length)

    def forward(self, x_t, x_ob, t):
        # define the hyperparameter:
        # diffusion step embedding
        step_embedding = self.step_encoder(t).unsqueeze(1).repeat(1, x_t.shape[1], 1)  # [b,2d] -> [b,l,2d]
        # add x_t and x_ob, and then element-wise product with step_embedding
        transformer_input = (x_ob + x_t) * step_embedding  # [b,l,2d] -> [z1,z2,...,zn], zi -> [b,1,2d]
        # add position encoding
        transformer_input_with_pos = self.positional_encoding(transformer_input)
        # model outputs
        transformer_output = self.transformer_encoder(transformer_input_with_pos)  # [h1,h2,...,hn] -> [b,l,2d]
        # predict x_0_hat
        return transformer_output  # [b,l,2d]
