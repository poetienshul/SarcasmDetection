import numpy as np
import json
from baselines import load_test_data, load_train_data

# Modified version of load_train_data for individual context
def load_train_data_individual(source):
    if source == 'reddit':
        filename = 'data/reddit_train.jsonl'
    elif source == 'twitter':
        filename = 'data/twitter_train.jsonl'

    with open(filename, encoding='utf8') as fh:
        data = [json.loads(jline) for jline in fh.read().strip().split('\n')]

    contexts = np.array([d['context'] for d in data])
    sentences = np.array([d['response'] for d in data])
    targets = np.array([1 if d['label']=='SARCASM' else 0 for d in data])
    return sentences, contexts, targets

# Modified version of load_test_data for individual context
def load_test_data_individual(source):
    if source == 'reddit':
        filename = 'data/reddit_test.jsonl'
    elif source == 'twitter':
        filename = 'data/twitter_test.jsonl'
    with open(filename, encoding='utf8') as fh:
        data = [json.loads(jline) for jline in fh.read().strip().split('\n')]

    contexts = np.array([d['context'] for d in data])
    sentences = np.array([d['response'] for d in data])
    return sentences, contexts

def generate_stats():
    source = 'twitter'
    train = load_train_data(source)
    test = load_test_data(source)
    train_samples = len(train[0])
    test_samples = len(test[0])
    print('Number of train samples: ', train_samples)
    print('Number of test samples: ', test_samples)
    sarcasm_samples = np.count_nonzero(train[2])
    print('Number SARCASM train samples: ', sarcasm_samples)
    print('Number NOT SARCASM train samples:', train_samples - sarcasm_samples)
    train_sentences = [s.split() for s in train[0]]
    test_sentences = [s.split() for s in test[0]]
    print('Average length of train sentence:', np.mean([len(s) for s in train_sentences]))
    print('Average length of test sentence:', np.mean([len(s) for s in test_sentences]))
    train_ind = load_train_data_individual(source)
    test_ind = load_test_data_individual(source)
    train_contexts_len = []
    test_contexts_len = []
    for contexts in train_ind[1]:
        for c in contexts:
            count = 0
            for w in c.split(' '):
                count += 1
            train_contexts_len.append(count)
    for contexts in test_ind[1]:
        for c in contexts:
            count = 0
            for w in c.split(' '):
                count += 1
            test_contexts_len.append(count)
    print('Average length of combined train context:', np.mean(train_contexts_len))
    print('Average length of combined test context:', np.mean(test_contexts_len))
    train_contexts = [s.split(' ') for s in train[1]]
    test_contexts = [s.split(' ') for s in test[1]]
    print('Average length of combined train context:', np.mean([len(s) for s in train_contexts]))
    print('Average length of combined test context:', np.mean([len(s) for s in test_contexts]))

if __name__ == "__main__":
    generate_stats()