import torch
import torch.nn as nn 
import math

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_in, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, d_in).float()
        pe.requires_grad = False

        for pos in range(max_len):
            for i in range(0, d_in, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_in)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_in)))
        
        # include the batch size
        self.pe = pe.unsqueeze(0)

    def forward (self, x):
        return self.pe
    

class BERTEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embed_size,
                 seq_len=64,
                 dropout=0.1
                 ):
        super().__init__()
        self.embed_size = embed_size
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbeddings(embed_size, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)

    