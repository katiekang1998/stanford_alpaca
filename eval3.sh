#!/bin/bash

# NAME=gsm8k_fft_full_constantlr


# for CHECKPOINT in 311 623 934 1246 1557 1869 2180 2492 2803 3110;
# do
#    echo ckpts/$NAME/checkpoint-$CHECKPOINT
#    python eval.py --ckpt_dir ckpts/$NAME/checkpoint-$CHECKPOINT
# done


# #!/bin/bash
# python eval.py --ckpt_dir ckpts/gsm8k_fft_full_constantlr/checkpoint-1557/
# python eval.py --ckpt_dir ckpts/gsm8k_fft_full_constantlr/checkpoint-934/






# NUM_TRAIN_POINTS=500
# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_mixed$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# NUM_TRAIN_POINTS=10000
# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type hard_skip --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_mixed10000 -maxdepth 2 -type f -name "*.safetensors" -delete



# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type hard_skip --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# NUM_TRAIN_POINTS=1000
# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type hard_full --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_hard_full$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_hard_full$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# NUM_TRAIN_POINTS=1000
# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type hard_full --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_hard_full$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_hard_full$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete





NUM_TRAIN_POINTS=30000

torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
ray stop
python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_mixed$NUM_TRAIN_POINTS
find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete


torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type easy_full --num_train_points $NUM_TRAIN_POINTS
ray stop
python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_easy_full$NUM_TRAIN_POINTS
find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_easy_full$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete


torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type hard_skip --num_train_points $NUM_TRAIN_POINTS
ray stop
python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS
find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete