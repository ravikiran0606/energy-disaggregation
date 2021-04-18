# LSTM with Attention

# Refrigerator
CUDA_VISIBLE_DEVICES=2 python3 train.py --model_type LSTM --appliance refrigerator --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256
CUDA_VISIBLE_DEVICES=2 python3 train.py --model_type LSTM --appliance refrigerator --window_segment_size 33 --num_epochs 20 --learning_rate 0.0001 --train_batch_size 256 --eval_batch_size 256

# Dishwaser
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type LSTM --appliance dishwaser --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type LSTM --appliance dishwaser --window_segment_size 33 --num_epochs 20 --learning_rate 0.0001 --train_batch_size 256 --eval_batch_size 256

# CNN:

# Refrigerator
CUDA_VISIBLE_DEVICES=2 python3 train.py --model_type CNN --appliance refrigerator --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256

# Dishwaser
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type CNN --appliance dishwaser --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256

