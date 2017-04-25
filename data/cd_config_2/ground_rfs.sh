#!/bin/bash

#produce grounded reward functions for L2 data and synced L0 data

script=/Users/edwardwilliams/documents/research/glamdp/plans/ground_rf.py

python $script --rf means/cd2_L0_synced_lifted.ml --domain 2 --out means/cd2_L0_synced_grounded.ml

python $script --rf ends/cd2_L2.ml --domain 2 --out ends/cd2_L2_grounded.ml
