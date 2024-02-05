import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
from dataloader import BERTDataset
from model import BERT, BERTLM, BERTTrainer



args = Namespace(
    
    VOCAB_SIZE = 30000,
    N_SEGMENTS = 3,
    MAX_LEN = 512,
    EMBED_DIM = 768,
    N_LAYERS = 12,
    ATTN_HEADS = 12,
    DROPOUT = 0.1,
    # Data and path information
    frequency_cutoff=25,
    data_path = "datasets/pairs.txt",
    model_state_file='model.pth', 
    review_csv='data/yelp/reviews_with_splits_lite.csv', 
    save_dir='model_storage/ch3/yelp/', 
    vectorizer_file='vectorizer.json',
    # No model hyperparameters
    # Training hyperparameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # Runtime options omitted for space
)



train_data = BERTDataset.load_data_and_get_tokenize(data_pair_path=args.data_path, seq_len=args.MAX_LEN)

tokenizer = train_data.tokenizer

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

# print(len(tokenizer.vocab))

bert_model = BERT(
  vocab_size=len(tokenizer.vocab),
  d_in=768,
  n_layers=2,
  n_heads=12,
  dropout=0.1
)

bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu')
# epochs = 20