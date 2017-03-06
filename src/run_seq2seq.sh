#!/bin/bash

nl=../data/lifted/L0.en
ml=../data/lifted/L0.ml

python train_seq2seq.py --natural_language $nl --machine_language $ml
