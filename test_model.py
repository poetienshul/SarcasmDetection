import random
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import dataloader

import time
import sys
# torch.set_printoptions(threshold=500000)
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class BERTSarcasm(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # print (text)
        # print (torch.max(text))
        # print (torch.min(text))
        # print (text.shape)
        # with torch.no_grad():
        #     embedded = self.bert(text)[0]
        embedded = self.bert(text)[0]
        # print ('embedded: {}'.format(embedded))
        # print ('embedded shape: {}'.format(embedded.shape))
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)

        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)

        # scores = F.log_softmax(output, dim=1)
        
        #output = [batch size, out dim]
        
        return output


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator):
        optimizer.zero_grad()
        # print ('batch: {}'.format(batch.sequence))
        predictions = model(batch.sequence)#.squeeze(1)
        # print ('predicionts shape: {}'.format(predictions.shape))
        # print ('ground truth shape: {}'.format(batch.label.shape))
        # print ('predictions: '.format(predictions))
        print (predictions)
        print (batch.label)
        loss = criterion(predictions, batch.label.float())
        # print ('batch train loss: {}'.format(loss))
        acc = binary_accuracy(predictions, batch.label.float())
        # print ('batch train acc: {}'.format(acc))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.sequence)#.squeeze(1)
            
            loss = criterion(predictions, batch.label.float())
            acc = binary_accuracy(predictions, batch.label.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def test_predictions(model, iterator):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in iterator:
            predictions.extend(model(batch.sequence))#.squeeze(1)
            
            
        
    return predictions

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# def predict_sentiment(model, tokenizer, sentence):
#     model.eval()
#     tokens = tokenizer.tokenize(sentence)
#     print (tokens)
#     tokens = tokens[:max_input_length-2]
#     indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
#     tensor = torch.LongTensor(indexed).to(device)
#     tensor = tensor.unsqueeze(0)
#     prediction = torch.sigmoid(model(tensor))
#     return prediction.item()
def create_submission(predictions, source, label_dict):
    # classes = {1: 'SARCASM', 0: 'NOT_SARCASM'}
    classes = {v: k for k, v in label_dict.items()}
    print (classes)
    with open(source + '_answer.txt', 'w') as fh:
        for i, p in enumerate(predictions):
            fh.write('{}_{}, {}\n'.format(source, i+1, classes[p]))

if __name__ == '__main__':
    bert = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = False
    DROPOUT = 0.25
    LEARNING_RATE = 0.2
    MOMENTUM=0.95

    model = BERTSarcasm(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 30

    best_valid_loss = float('inf')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    special_tokens_dict = {'eos_token': '[EOS]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print (len(tokenizer))
    model.bert.resize_token_embeddings(len(tokenizer))
    
    train_iterator, valid_iterator, test_iterator, label_dict = dataloader.fetch_data('reddit', BATCH_SIZE=BATCH_SIZE, SEED=SEED)

    # torch.save(model.state_dict(), 'tut6-model.pt')
    model.load_state_dict(torch.load('tut6-model.pt'))

    test_predictions = test_predictions(model, test_iterator)

    # print (test_predictions)
    print (torch.stack(test_predictions).shape)
    test_predictions = torch.argmax(torch.stack(test_predictions), axis=1).cpu().numpy()
    create_submission(test_predictions, 'reddit', label_dict)
    # model.load_state_dict(torch.load('tut6-model.pt'))

    # test_loss, test_acc = evaluate(model, test_iterator, criterion)

    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    # predict_sentiment(model, tokenizer, "This film is terrible")