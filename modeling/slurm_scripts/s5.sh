#!/bin/bash
python3 train.py --model_type LSTM --appliance refrigerator --lstm_num_layers 1 --window_segment_size 65 --num_epochs 20 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256