#!/bin/bash

nl_train=../data/lifted_merged/no_l0/no_l0_train.en
ml_train=../data/lifted_merged/no_l0/no_l0_train.ml
nl_test=../data/lifted_merged/no_l0/no_l0_test.en
ml_test=../data/lifted_merged/no_l0/no_l0_test.ml

model_path=trained_models/trained_seq2seq_3_6_17

python run_seq2seq.py --nl_train $nl_train --ml_train $ml_train --nl_test $nl_test --ml_test $ml_test --save $model_path
