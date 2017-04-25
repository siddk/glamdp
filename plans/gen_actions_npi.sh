#!/bin/bash
data=../data/npi_train_test/L0_action_sequences

python segments_to_actions.py --npi_segments $data/L0_test_actions.ml --action_trajectories $data/trajectories_gt.txt
