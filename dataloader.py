import torch
# import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import itertools
from berttokenizer import tokenizer_trainer
from transformers import BertTokenizer
# import torch.nn.functional as F


class BERTDataset(Dataset):
    def __init__(self, 
                 data_pair,
                 tokenizer,
                 seq_len
                 ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
    
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
         # Adding PAD token for labels
        t1 = [self.tokenizer.vocab["[CLS]"]] + t1_random + [self.tokenizer.vocab["[SEP]"]]
        t2 = t2_random + [self.tokenizer.vocab["[SEP]"]]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len   
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1+t2)[:self.seq_len]
        bert_label = (t1_label+t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label
        }

        return {key: torch.tensor(value) for key,value in output.items()}



    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        for i, token in enumerate(tokens):
            prob = random.random()

            #remove cls and sep tokens (beginning and end of sentence tokens)

            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab["[MASK]"])
                
                elif prob <0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                else:
                    output.append(token_id)
                
                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)


        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label





    def get_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return t1, self.get_random_line(), 0
        else:
            return t1, t2, 1

    def get_corpus_line(self, index):
        return self.lines[index][0], self.lines[index][1]

    def get_random_line(self):
        return self.lines[random.randrange(self.corpus_lines)][1]


    @classmethod
    def load_data_and_get_tokenize(cls, data_pair_path, seq_len):
        # load pairs from disk
        data_pair = []
        with open(data_pair_path, 'r') as f:
            for line in f:
                data_pair.append(line.strip().split('\t'))

        try:
            tokenizer = BertTokenizer.from_pretrained('./tokenizer/tokenizer-1-vocab.txt', local_files_only=True)
        except:
            tokenizer = tokenizer(data_pair)
            tokenizer.train()
            tokenizer = tokenizer.get_tokenizer()

        return cls(data_pair, tokenizer, seq_len)
    

