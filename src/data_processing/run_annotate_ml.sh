#!/bin/bash

datalocation=../../data/rss_data_new

python ml_to_actions.py --means $datalocation/ends/L2.ml --traces $datalocation/traces_RSS.txt --out $datalocation/L2.at - 
