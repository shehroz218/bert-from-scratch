bert_model = BERT(
#   vocab_size=len(tokenizer.vocab),
#   d_in=768,
#   n_layers=2,
#   n_heads=12,
#   dropout=0.1
# )

# bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
# bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu')
# epochs = 20