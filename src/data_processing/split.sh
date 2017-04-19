#!/bin/bash


dat=../../data/rss_data

#split ends data
python split_data.py --ml $dat/ends/L2.ml --en $dat/ends/L2.en --pct 0.8 --out $dat/ends/L2 --pfx E
#split means data
python split_data.py --ml $dat/means/L0.ml --en $dat/means/L0.en --pct 0.8 --out $dat/means/L0 --pfx M
