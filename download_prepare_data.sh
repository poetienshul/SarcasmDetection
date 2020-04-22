#!/bin/bash
echo "Downloading Data..."
if ! test -f "reddit_training.zip"; then
    wget https://github.com/EducationalTestingService/sarcasm/releases/download/v1.0/reddit_training.zip
fi
if ! test -f "twitter_training.zip"; then
    wget https://github.com/EducationalTestingService/sarcasm/releases/download/v1.0/twitter_training.zip
fi
if ! test -f "reddit_test.zip"; then
    wget https://github.com/EducationalTestingService/sarcasm/releases/download/v2.0/reddit_test.zip
fi
if ! test -f "twitter_test.zip"; then
    wget https://github.com/EducationalTestingService/sarcasm/releases/download/v2.0/twitter_test.zip
fi

echo "Preparing Data..."
mkdir data
unzip '*_training.zip'
mv reddit/sarcasm_detection_shared_task_reddit_training.jsonl data/reddit_train.jsonl
mv twitter/sarcasm_detection_shared_task_twitter_training.jsonl data/twitter_train.jsonl
unzip '*_test.zip' -d data/

echo "Preparing Bert Data"
mkdir bert_data
python utils/transform_dataset.py --sep "[SEP]" --source_dir data --dest_dir bert_data

echo "Preparing Roberta Data"
mkdir roberta_data
python utils/transform_dataset.py --sep "</s> </s>" --source_dir data --dest_dir roberta_data

echo "Preparing XLNet Data"
mkdir xlnet_data
python utils/transform_dataset.py --sep "<SEP>" --source_dir data --dest_dir xlnet_data

rm -r __MACOSX/
rm -r reddit/
rm -r twitter/