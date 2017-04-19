#!/bin/bash

means=../../data/rss_data/means
ends=../../data/rss_data/ends
both=../../data/rss_data/both

#combine training data
python means_ends_dataset.py --means $means/L0_train --ends $ends/L2_train --out $both/all_train

#combine test data
python means_ends_dataset.py --means $means/L0_test --ends $ends/L2_test --out $both/all_test
