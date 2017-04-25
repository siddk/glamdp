#!/bin/bash

data=../data/npi_train_test/permuted_ends

python randomize_grounding.py --lifted $data/L2_test_lifted_gt.ml --out $data/randomized
