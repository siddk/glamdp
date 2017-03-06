#!/bin/bash

folder=../../data/lifted
out=../../data/lifted_merged/all/all

python create_train_test.py --in_folder $folder --pct_train 0.8 --out_folder $out
