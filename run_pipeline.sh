#!/bin/bash

#shared file paths
classifier_train=data/combined_rss_data/combined/train
classifier_test=data/combined_rss_data/combined/test
pred_means=predictions/pred_means
pred_ends=predictions/pred_ends
npi_train=data/combined_rss_data/ends/no_l0_train
npi_test=data/combined_rss_data/ends/no_l0_test
means_train=data/combined_rss_data/means/means_train

#these paths are also hard-coded in the experiment file.
ccgsettrc=predictions/predicted_means.ccgsettrc
settrc=predictions/ccg_train.settrc
#TODO: relative paths/move navi into this repo?
pipeline_exp=/home/edwardwilliams/research/navi/experiments/cleanup/cleanup_pipeline.exp
navi_path=/home/edwardwilliams/research/navi/dist/navi-1.2.jar

#logs
rf_results=results/rf_results.txt

#separating into separate scripts enables intermediate data products to be saved

#python predict_class.py --train $classifier_train --test $classifier_test --means $pred_means --ends $pred_ends
#python predict_rf.py --train $npi_train --test $npi_test --pred_ends $pred_ends --results $rf_results
python gen_settrc.py --parallel_data $means_train --out $settrc
python gen_settrc.py --parallel_data $pred_means --out $ccgsettrc --test
#java -Xmx8g -jar $navi_path $pipeline_exp
