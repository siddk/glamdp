#!/bin/bash


dat=../../data/rss_data
out=../../data/npi_train_test/L2_

#split ends data
python split_data.py --grounded_ml $dat/ends/L2.ml --lifted_ml $dat/ends/L2-lifted.ml --en $dat/ends/L2.en --pct 0.8 --out $out --pct 0.9
