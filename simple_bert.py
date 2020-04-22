import random
import numpy as np
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, XLNetForSequenceClassification, AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import dataloader

import argparse

import time
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch, model, train_iterator, optimizer, verbose=False):
    model.train()
    for i, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        loss, logits = model(batch.sequence, labels=batch.label.squeeze(1))
        loss.backward()
        optimizer.step()
        if verbose:
            print ("batch {} / {}: train loss: {}".format(i, num_batches, loss))

def evaluate(model, iterator):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        epoch_acc = 0
        for batch in iterator:
            loss, logits = model(batch.sequence, labels=batch.label.squeeze(1))

            preds = torch.argmax(logits, axis=1)
            correct = (preds == batch.label.squeeze(1)).float()
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test(model, iterator):
    predictions = []
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        epoch_acc = 0
        for batch in iterator:
            logits = model(batch.sequence)
            preds = torch.argmax(logits[0], axis=1)
            for p in preds:
                predictions.append(p.item())
    return predictions

def create_submission(predictions, source, label_dict):
    # classes = {1: 'SARCASM', 0: 'NOT_SARCASM'}
    classes = {v: k for k, v in label_dict.items()}
    print (classes)
    with open(source + '_answer.txt', 'w') as fh:
        for i, p in enumerate(predictions):
            fh.write('{}_{}, {}\n'.format(source, i+1, classes[p]))
    print ("Predictions written to {}_answer.txt".format(source))

def main(params):
    # Load pretrained specified model
    if params['model'] == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    elif params['model'] == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    elif params['model'] == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
    model = model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=1e-1)
    # optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    optimizer = AdamW(model.parameters(), lr=params['lr'], correct_bias=False, weight_decay=params['weight_decay'])

    d = dataloader.Dataloader(params)
    train_iterator, valid_iterator, test_iterator, label_dict = d.fetch_data()
    print ("# of train batches: {}".format(len(train_iterator)))
    print ("# of valid batches: {}".format(len(valid_iterator)))
    print ("# of test batches: {}".format(len(test_iterator)))

    if params['predict']:
        print ('Loading model from {}'.format(params['pretrained_model']))
        model.load_state_dict(torch.load(params['pretrained_model']))
        predictions = test(model, test_iterator)
        create_submission(predictions, params['data_source'], label_dict)
        sys.exit()

    best_valid_loss = float('inf')
    num_batches = len(train_iterator)

    for epoch in range(params['epochs']):
        train(epoch, model, train_iterator, optimizer, verbose=False)

        train_loss, train_acc = evaluate(model, train_iterator)
        # print ("epoch {}: train loss: {0:.3f}, acc: {0:.3f}".format(epoch, train_loss, train_acc))

        valid_loss, valid_acc = evaluate(model, valid_iterator)
        print ("Epoch {0}: train loss: {1:.3f}, acc: {2:.3f}; valid loss: {3:.3f}, acc: {4:.3f}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))
        
        if valid_loss < best_valid_loss:
            print ("Better valid loss found: {}, saving model to {}".format(valid_loss, params['save_model']))
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), params['save_model'])
        # torch.save(model.state_dict(), 'e_' + str(epoch) + params['save_model'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--seed', default=11171, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model', default='bert', choices=['bert', 'roberta', 'xlnet'])
    parser.add_argument('--data_source', default='twitter')
    parser.add_argument('--data_dir', default='bert_data')
    parser.add_argument('--save_model', default='simple_bert.pt')
    parser.add_argument('--predict', default=False, type=bool)
    parser.add_argument('--pretrained_model', default='simple_bert.pt')
    args = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    params = parse_args()
    main(params)
# roberta
# CUDA_VISIBLE_DEVICES=1 python simple_bert.py --model roberta --data_dir roberta_data --save_model simple_roberta.pt
# CUDA_VISIBLE_DEVICES=0 python simple_bert.py --model roberta --data_dir roberta_data --predict True --pretrained_model simple_roberta.pt

# xlnet
# CUDA_VISIBLE_DEVICES=1 python simple_bert.py --model xlnet --data_dir xlnet_data --save_model simple_xlnet.pt
# CUDA_VISIBLE_DEVICES=0 python simple_bert.py --model xlnet --data_dir xlnet_data --predict True --pretrained_model simple_xlnet.pt
