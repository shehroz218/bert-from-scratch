import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, heads, dropout=0.1):
        super().__init__()
        assert d_in % heads == 0

        self.d_k = d_in // heads
        self.d_in = d_in

        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_in, d_in)
        self.W_k = nn.Linear(d_in, d_in)
        self.W_v = nn.Linear(d_in, d_in)
        self.W_o = nn.Linear(d_in, d_in)

    def forward(self, x, mask=None):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = query @key.transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))

        scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = weights @ value

        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        return self.W_o(context)
    
class FeedForward(nn.Module):
    def __init__(self, d_in, d_middle, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(d_in, d_middle)
        self.fc2 = nn.Linear(d_middle, d_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_in=768,
                 heads=12,
                 feed_forward_hidden = 768*4,
                 dropout=0.1):
        super().__init__()

        self.layernorm = nn.LayerNorm(d_in)
        self.self_multihead = MultiHeadedAttention(d_in, heads, dropout)
        self.feed_forward = FeedForward(d_in, feed_forward_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, mask):
        x = self.self_multihead(embedding, mask)
        x = self.dropout(x)
        x = self.layernorm(x + embedding)
        feeed_forward_out = self.dropout(self.feed_forward(x))
        encoded = self.layernorm(x + feeed_forward_out)
        return encoded
