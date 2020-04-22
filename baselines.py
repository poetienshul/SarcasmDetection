import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import scipy
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# random.seed(1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_train_data(source):
    if source == 'reddit':
        filename = 'data/reddit_train.jsonl'
    elif source == 'twitter':
        filename = 'data/twitter_train.jsonl'

    with open(filename, encoding='utf8') as fh:
        data = [json.loads(jline) for jline in fh.read().strip().split('\n')]

    # sentences = np.array([d['response'] for d in data])
    # contexts = np.array([d['context'] for d in data])
    # targets = np.array([1 if d['label']=='SARCASM' else 0 for d in data])

    # num_contexts = 0
    # sum_contexts = 0
    # for d in data:
    # 	for c in d['context']:
    # 		words = c.split()
    # 		sum_contexts += len(words)
    # 		num_contexts += 1
    # print("average len of individual contexts: " + str(sum_contexts * 1.0 / num_contexts))

    sentences = np.array([d['response'] for d in data])
    contexts = np.array([' '.join(d['context']) for d in data])
    targets = np.array([1 if d['label']=='SARCASM' else 0 for d in data])
    return sentences, contexts, targets

def load_test_data(source):
    if source == 'reddit':
        filename = 'data/reddit_test.jsonl'
    elif source == 'twitter':
        filename = 'data/twitter_test.jsonl'
    with open(filename, encoding='utf8') as fh:
        data = [json.loads(jline) for jline in fh.read().strip().split('\n')]

    # sentences = np.array([d['response'] for d in data])
    # contexts = np.array([d['context'] for d in data])

    num_contexts = 0
    sum_contexts = 0
    for d in data:
    	for c in d['context']:
    		words = c.split()
    		sum_contexts += len(words)
    		num_contexts += 1
    print("average len of individual contexts: " + str(sum_contexts * 1.0 / num_contexts))

    sentences = np.array([d['response'] for d in data])
    contexts = np.array([' '.join(d['context']) for d in data])
    return sentences, contexts

def create_submission(predictions, source):
    classes = {1: 'SARCASM', 0: 'NOT_SARCASM'}
    with open(source + '_answer.txt', 'w') as fh:
        for i, p in enumerate(predictions):
            fh.write('{}_{}, {}\n'.format(source, i+1, classes[p]))

# for source in ['reddit', 'twitter']:
for source in ['reddit']:
    train_sentences, train_contexts, train_targets = load_train_data(source)
    test_sentences, test_contexts = load_test_data(source)

    # print("len(train_sentences): " + str(len(train_sentences)))
    # print("len(test_sentences): " + str(len(test_sentences)))
    # print(np.count_nonzero(train_targets == 1))

    # sentence_sum = 0
    # for sentence in train_sentences:
    # 	words = sentence.split()
    # 	# print(sentence, len(sentence))
    # 	sentence_sum += len(words)
    # print("Average length of Response: " + str(sentence_sum * 1.0/ len(train_sentences)))

    # sentence_sum = 0
    # for sentence in test_sentences:
    # 	words = sentence.split()
    # 	# print(words)
    # 	# print(sentence, len(sentence))
    # 	sentence_sum += len(words)
    # print("Average length of Response: " + str(sentence_sum * 1.0/ len(test_sentences)))

    # context_sum = 0
    # for context in train_contexts:
    # 	print(context)
    # 	words = context.split()
    # 	context_sum += len(words)
    # print("Average length of Context: " + str(context_sum * 1.0/ len(train_contexts)))

    # context_sum = 0
    # for context in test_contexts:
    # 	words = context.split()
    # 	context_sum += len(words)
    # print("Average length of Context: " + str(context_sum * 1.0/ len(test_contexts)))

    count_vect = CountVectorizer()
    X_context_counts = count_vect.fit_transform(train_contexts)
    X_sent_counts = count_vect.transform(train_sentences)

    tfidf_transformer = TfidfTransformer()
    X_context_tfidf = tfidf_transformer.fit_transform(X_context_counts)
    X_sent_tfidf = tfidf_transformer.transform(X_sent_counts)

    train_input = scipy.sparse.hstack((X_context_tfidf, X_sent_tfidf))


    clf = RandomForestClassifier().fit(train_input, train_targets)
    # clf = MultinomialNB().fit(train_input, train_targets)
    # clf = svm.SVC().fit(train_input, train_targets)

    X_context_counts = count_vect.transform(test_contexts)
    X_sent_counts = count_vect.transform(test_sentences)

    X_context_tfidf = tfidf_transformer.transform(X_context_counts)
    X_sent_tfidf = tfidf_transformer.transform(X_sent_counts)
    test_input = scipy.sparse.hstack((X_context_tfidf, X_sent_tfidf))
    create_submission(clf.predict(test_input), source)
