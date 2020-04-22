import random
import numpy as np
from transformers import BertTokenizer, BertModel, BertForTokenClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import dataloader

import time
import sys
# torch.set_printoptions(threshold=500000)
SEED = 135

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


class BERTSarcasm(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout, vocab_size):
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        #####################################
        # print ('embedding_dim: {}'.format(embedding_dim))
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.n_layers = n_layers
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, text):
        # with torch.no_grad():
        # embedded = self.bert(text)[0]
        # _, hidden = self.rnn(embedded)
        # if self.rnn.bidirectional:
        #     hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # else:
        #     hidden = self.dropout(hidden[-1,:,:])
        # output = self.out(hidden)
        # scores = F.log_softmax(output, dim=1)
        # print ("input size: {}".format(text.shape))
        # print ('len(text) : {}'.format(len(text)))
        embedded_out = self.embedding(text)
        # print ('embedded shape: {}'.format(embedded_out.view(self.embedding_dim, len(text), -1).shape))
        
        # lstm_out, lstm_hidden = self.lstm(embedded_out.view(len(text), 1, -1))
        lstm_out, _ = self.lstm(embedded_out.view(text.shape[1], len(text), -1))
        # print (lstm_hidden)
        # print (lstm_out.shape)
        # print (lstm_hidden.shape)
        # print ('lstm_hidden: {}'.format(lstm_hidden.view(self.n_layers, len(text), self.hidden_dim).shape))
        # print ('lstm_hidden: {}'.format(lstm_hidden.permute(1, 0, 2).reshape(len(text), -1).shape))
        # outs = self.linear(lstm_hidden.permute(1, 0, 2).reshape(len(text), -1))
        # print ('lstm_out: {}'.format(lstm_out[-1,:,:].shape))
        outs = self.linear(lstm_out[-1,:,:])
        # print ('outs: {}'.format(outs.shape))
        scores = F.log_softmax(outs, dim=1)

        return scores


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    # rounded_preds = torch.round(torch.sigmoid(preds))
    rounded_preds = torch.argmax(preds, axis=1)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    # for batch in tqdm(iterator):
    for batch in iterator:
        optimizer.zero_grad()
        # print ('batch: {}'.format(batch.sequence))
        predictions = model(batch.sequence)#.squeeze(1)
        # print ('predicionts shape: {}'.format(predictions.shape))
        # print ('ground truth shape: {}'.format(batch.label.squeeze(1).shape))
        # print (predictions)
        # print (F.one_hot(batch.label.squeeze(1)))
        # print ('predictions: '.format(predictions))
        # print (predictions)
        # print (batch.label)
        # print (predictions.shape)
        # print (batch.label.shape)
        loss = criterion(predictions, batch.label.squeeze(1))
        # print ('batch train loss: {}'.format(loss))
        acc = binary_accuracy(predictions, batch.label.squeeze(1))
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
            
            # loss = criterion(predictions, batch.label.float())
            loss = criterion(predictions, batch.label.squeeze(1))
            # print ('batch train loss: {}'.format(loss))
            acc = binary_accuracy(predictions, batch.label.squeeze(1))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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


if __name__ == '__main__':
    # bert = BertForTokenClassification.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2
    N_LAYERS = 1
    BIDIRECTIONAL = False
    DROPOUT = 0.25
    LEARNING_RATE = 1e-5
    MOMENTUM = 0.95
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    special_tokens_dict = {'eos_token': '[EOS]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print (len(tokenizer))
    bert.resize_token_embeddings(len(tokenizer))

    model = BERTSarcasm(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, len(tokenizer))

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 2000

    best_valid_loss = float('inf')

    
    train_iterator, valid_iterator, test_iterator, label_dict = dataloader.fetch_data('reddit', BATCH_SIZE=BATCH_SIZE, SEED=SEED)
    print (len(train_iterator))
    print (len(valid_iterator))

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            
        end_time = time.time()
            
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    # model.load_state_dict(torch.load('tut6-model.pt'))

    # test_loss, test_acc = evaluate(model, test_iterator, criterion)

    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    # predict_sentiment(model, tokenizer, "This film is terrible")