import torch
import torch.nn as nn
from embed import BERTEmbedding
from encoder import EncoderLayer

class BERT(nn.Module):
    def __init__(self, 
                vocab_size, 
                d_in=768, 
                n_layers=12, 
                n_heads=12, 
                dropout=0.1):
        super().__init__()

        self.d_in = d_in
        self.n_layers = n_layers
        self.heads = n_heads

        #paper has 4*hidden_size for ff_hidden_size
        self.feed_forward_hidden = 4*d_in

        self.embedding = BERTEmbedding(vocab_size, d_in)

        #multi attention
        self.encoder_block = nn.ModuleList(
            [EncoderLayer(d_in, n_heads, d_in*4, dropout) for _ in range(n_layers)]
        )
    
    def forward(self, x, segment_info):
        mask = (x>0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for layer in self.encoder_block:
            x = layer(x, mask)
        return x
    

class NextSentencePrediction(torch.nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    
class BERTLM(nn.Module):

    def __init__(self, bert, vocab_size):

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(bert.d_in)
        self.mask_lm = MaskedLanguageModel(bert.d_in, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)