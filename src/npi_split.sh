#!/bin/bash

script=data_processing/npi_L0_train_test.py
seg_actions=../data/segmented_rss/ALL_SEGMENTED_ACTIONS.ml
seg_en=../data/segmented_rss/ALL_SEGMENTED.en
seg_rf=../data/segmented_rss/ALL_SEGMENTED_RF.ml
unseg_en=../data/segmented_rss/UNSEGMENTED.en
unseg_rf=../data/segmented_rss/UNSEGMENTED_RF.ml
out=../data/npi_train_test/


python $script --seg_actions $seg_actions --seg_en $seg_en --seg_rf $seg_rf --unseg_en $unseg_en --unseg_rf $unseg_rf --npi_test_frac 0.3 --out $out
