import sys
sys.path.append('../')

import os
from pathlib import Path
import torch
import torch.nn as nn
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
import shutil


class tokenizer_trainer():
    def __init__(self, data_pair):
        self.pairs = data_pair
        self.paths = self._get_paths()
        self.tokenizer = BertWordPieceTokenizer(
                                clean_text=True,
                                handle_chinese_chars=False,
                                strip_accents=False,
                                lowercase=True)
        
    def _get_paths(self):
        os.mkdir('./data')
        text_data = []
        file_count = 0

        for sample in tqdm.tqdm([x[0] for x in self.pairs]):
            text_data.append(sample)

            # once we hit the 10K mark, save to file, reset text_data and increment file_count
            if len(text_data) == 10000:
                with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1

        paths = [str(x) for x in Path('./data').glob('**/*.txt')]
        return paths
    
    def _clean_directory(self):
        shutil.rmtree('./data')
        
    def train(self,
            vocab_size=30_000, 
            min_frequency=5, 
            limit_alphabet=1000, 
            wordpieces_prefix='##', 
            special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']):
        

        self.tokenizer.train( 
            files=self.paths,
            vocab_size=vocab_size, 
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet, 
            wordpieces_prefix=wordpieces_prefix,
            special_tokens=special_tokens
            )
        
        os.mkdir('../tokenizer')
        self.tokenizer.save_model('../tokenizer', 'tokenizer-1')
        
        self._clean_directory()

    def get_tokenizer(self):
        return self.tokenizer



# BertTokenizer.from_pretrained('./tokenizer/tokenizer-1-vocab.txt', local_files_only=True)

