import torch
from torchtext import datasets
from torchtext import data
from torchtext.data import TabularDataset
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
import random

class Dataloader:
    def __init__(self, params):
        self.source = params['data_source']
        self.data_dir = params['data_dir']
        self.batch_size = params['batch_size']
        self.seed = params['seed']
        self.model = params['model']

        if self.model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        elif self.model == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.max_input_length = self.tokenizer.max_model_input_sizes['roberta-base']
        elif self.model == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            # self.max_input_length = self.tokenizer.max_model_input_sizes['xlnet-base-cased']
            self.max_input_length = 512
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) + 2 < self.max_input_length:
            return tokens
        else:
            return [tokens[0]] + tokens[len(tokens) - self.max_input_length + 3:]

    def fetch_data(self):
        init_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        eos_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token
        unk_token = self.tokenizer.unk_token

        init_token_idx = self.tokenizer.convert_tokens_to_ids(init_token)
        sep_token_idx = self.tokenizer.convert_tokens_to_ids(sep_token)
        eos_token_idx = self.tokenizer.convert_tokens_to_ids(eos_token)
        pad_token_idx = self.tokenizer.convert_tokens_to_ids(pad_token)
        unk_token_idx = self.tokenizer.convert_tokens_to_ids(unk_token)

        TEXT = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = self.tokenize_and_cut,
                        preprocessing = self.tokenizer.convert_tokens_to_ids,
                        init_token = init_token_idx,
                        eos_token = eos_token_idx,
                        pad_token = pad_token_idx,
                        unk_token = unk_token_idx)

        LABEL = data.Field(batch_first=True, pad_token=None, unk_token=None)

        train_datafields = [('sequence', TEXT), ('label', LABEL)]

        all_train_data = TabularDataset.splits(
            path=self.data_dir,
            train=self.source+'_train.tsv',
            format='tsv',
            skip_header=True,
            fields=train_datafields)[0]


        test_datafields = [('sequence', TEXT)]

        test_data = TabularDataset.splits(
            path=self.data_dir,
            train=self.source+'_test.tsv',
            format='tsv',
            skip_header=True,
            fields=test_datafields)[0]

        train_data, valid_data = all_train_data.split(split_ratio=0.9, random_state=random.seed(self.seed))

        LABEL.build_vocab(train_data)
        print (LABEL.vocab.stoi)

        train_iterator, valid_iterator = data.Iterator.splits(
            (train_data, valid_data), 
            batch_size = self.batch_size, 
            sort=False,
            shuffle=True,
            device = self.device)
        
        test_iterator = data.Iterator(test_data, batch_size=self.batch_size, device=self.device, train=False, sort=False)
        return train_iterator, valid_iterator, test_iterator, LABEL.vocab.stoi
