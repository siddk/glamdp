#!/bin/bash

datpath=../../data/

python create_classifier_data.py --lo $datpath/grounded/L0.en --hi $datpath/grounded/L1.en $datpath/grounded/L2.en --out $datpath/language_classifier/classifier --split 0.8
