#!/bin/bash

if [ "$1" = "train-pre" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.es --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.es --vocab=vocab.json --pretrain-model=./model_output/model.bin --batch-size=4 --cuda
elif [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.es --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.es --vocab=vocab.json --cuda
elif [ "$1" = "traincpu" ]; then
	python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.es --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.es --vocab=vocab.json
elif [ "$1" = "test" ]; then
  CUDA_VISIBLE_DEVICES=0 python run.py decode ./model_output/model.bin ./en_es_data/test.en ./en_es_data/test.es outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.es --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.es --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.en ./en_es_data/test.es outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.es vocab.json
else
	echo "Invalid Option Selected"
fi
