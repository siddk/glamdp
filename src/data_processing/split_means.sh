#!/bin/bash


dat=../../data/combined_rss_data/means

python split_data.py --ml $dat/all_means_actions.ml --en $dat/all_means.en --pct 0.8 --out $dat/means
