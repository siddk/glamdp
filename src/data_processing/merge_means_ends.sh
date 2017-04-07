#!/bin/bash

means=../../data/combined_rss_data/means
ends=../../data/combined_rss_data/ends
combined=../../data/combined_rss_data/combined

#combine training data
python means_ends_dataset.py --means $means/means_train --ends $ends/no_l0_train --out $combined/train

#combine test data
python means_ends_dataset.py --means $means/means_test --ends $ends/no_l0_test --out $combined/test
