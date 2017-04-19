#!/bin/bash

#runs training and evaluation on grounded RNN for both means and ends data
dat=data/rss_data

python predict_rf_rnn.py --train $dat/both/all_train --test $dat/both/all_test --results $dat/grounded_logs/all.txt

python predict_rf_rnn.py --train $dat/both/all_train --test $dat/ends/L2_test --results $dat/grounded_logs/ends_test.txt

python predict_rf_rnn.py --train $dat/both/all_train --test $dat/means/L0_test --results $dat/grounded_logs/means_test.txt
