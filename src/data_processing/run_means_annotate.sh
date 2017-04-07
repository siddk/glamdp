#!/bin/bash

datalocation=../../data/combined_rss_data

python annotate_means.py --means $datalocation/means.ml --traces $datalocation/traces_means.txt --out $datalocation/means_actions.ml
