import sys
import json
import argparse

def main(params):
    for filename in ['reddit_train', 'twitter_train']:
        with open(params['source_dir'] + '/' + filename + '.jsonl', encoding="utf8") as fh:
            data = [json.loads(jline) for jline in fh.read().strip().split('\n')]

        with open(params['dest_dir'] + '/' + filename + '.tsv', 'w', encoding="utf8") as fh:
            fh.write('sequence\tlabel\n')
            for d in data:
                fh.write('{} {} {}\t{}\n'.format(' '.join(d['context']), params['sep'], d['response'], d['label']))

    for filename in ['reddit_test', 'twitter_test']:
        with open(params['source_dir'] + '/' + filename + '.jsonl', encoding="utf8") as fh:
            data = [json.loads(jline) for jline in fh.read().strip().split('\n')]

        with open(params['dest_dir'] + '/' + filename + '.tsv', 'w', encoding="utf8") as fh:
            fh.write('sequence\n')
            for d in data:
                fh.write('{} {} {}\n'.format(' '.join(d['context']), params['sep'], d['response']))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sep', default='[SEP]')
    parser.add_argument('--source_dir', default='data')
    parser.add_argument('--dest_dir', default='bert_data')
    args = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    params = parse_args()
    main(params)
