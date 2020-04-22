# SarcasmDetection
To install python3.7 requirements:  

    pip install -r requirements.txt

First download the competition data & format it for our models  

    ./download_prepare_data.sh

Then train and generate predictions using one of the following models:  

### bert
    python simple_bert.py --data_source twitter --model bert --data_dir bert_data --save_model simple_bert.pt
    python simple_bert.py --data_source twitter --model bert --data_dir bert_data --predict True --pretrained_model simple_bert.pt

### roberta
    python simple_bert.py --data_source twitter --model roberta --data_dir roberta_data --save_model simple_roberta.pt
    python simple_bert.py --data_source twitter --model roberta --data_dir roberta_data --predict True --pretrained_model simple_roberta.pt

### xlnet
    python simple_bert.py --data_source twitter --model xlnet --data_dir xlnet_data --save_model simple_xlnet.pt
    python simple_bert.py --data_source twitter --model xlnet --data_dir xlnet_data --predict True --pretrained_model simple_xlnet.pt

Generating predictions will generate a file like *twitter_answer.txt*, convert to .zip by:  

    ./create_submission.sh