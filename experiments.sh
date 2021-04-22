# LSTM with Attention

# Refrigerator
CUDA_VISIBLE_DEVICES=2 python3 train.py --model_type LSTM --appliance refrigerator --lstm_num_layers 2 --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256

# Dishwaser
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type LSTM --appliance dishwaser --lstm_num_layers 2 --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256

# CNN:

# Refrigerator
CUDA_VISIBLE_DEVICES=2 python3 train.py --model_type CNN --appliance refrigerator --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256

# Dishwaser
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_type CNN --appliance dishwaser --window_segment_size 3 --num_epochs 10 --learning_rate 1e-5 --train_batch_size 256 --eval_batch_size 256

# Refrigerator: Try window_segment_size 21, 51, 31, 41
# Dishwaser: Try window_segment_size  21, 51, 31, 41


# Experiments by Ravi

srun -u --gres=gpu:1 --nodelist ink-gary slurm_scripts/s1.sh
srun -u --gres=gpu:1 --nodelist ink-gary slurm_scripts/s2.sh
srun -u --gres=gpu:1 --nodelist ink-gary slurm_scripts/s3.sh
srun -u --gres=gpu:1 --nodelist ink-gary slurm_scripts/s4.sh
srun -u --gres=gpu:1 --nodelist ink-gary slurm_scripts/s5.sh

srun -u --gres=gpu:1 --nodelist ink-lisa slurm_scripts/s6.sh
srun -u --gres=gpu:1 --nodelist ink-lisa slurm_scripts/s7.sh
srun -u --gres=gpu:1 --nodelist ink-lisa slurm_scripts/s8.sh
srun -u --gres=gpu:1 --nodelist ink-lisa slurm_scripts/s9.sh
srun -u --gres=gpu:1 --nodelist ink-lisa slurm_scripts/s10.sh


CUDA_VISIBLE_DEVICES=3 bash slurm_scripts/s1.sh
CUDA_VISIBLE_DEVICES=4 bash slurm_scripts/s2.sh
CUDA_VISIBLE_DEVICES=5 bash slurm_scripts/s3.sh

CUDA_VISIBLE_DEVICES=6 bash slurm_scripts/s6.sh
CUDA_VISIBLE_DEVICES=0 bash slurm_scripts/s7.sh


