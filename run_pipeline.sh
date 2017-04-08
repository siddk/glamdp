#!/bin/bash

#shared file paths
classifier_train=data/combined_rss_data/combined/train
classifier_test=data/combined_rss_data/combined/test
pred_means=predictions/pred_means
pred_ends=predictions/pred_ends
npi_train=data/combined_rss_data/ends/no_l0_train
npi_test=data/combined_rss_data/ends/no_l0_test
ccgsettrc=predictions/predicted_means.ccgsettrc

#logs
rf_results=results/rf_results.txt

#separating into separate scripts enables intermediate data products to be saved

#python predict_class.py --train $classifier_train --test $classifier_test --means $pred_means --ends $pred_ends
python predict_rf.py --train $npi_train --test $npi_test --pred_ends $pred_ends --results $rf_results
#python gen_ccg.py --pred_means $pred_means --ccg $ccgsettrc
