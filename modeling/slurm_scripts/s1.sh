#!/bin/bash
python3 train.py --model_type CNN --appliance kitchen_outlets --window_segment_size 11 --num_epochs 20 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256