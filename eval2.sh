#!/bin/bash

# NAME=gsm8k_fft_hard50_constantlr


# for CHECKPOINT in 156 312 468 624 780;
# do
#    echo ckpts/$NAME/checkpoint-$CHECKPOINT
#    python eval.py --ckpt_dir ckpts/$NAME/checkpoint-$CHECKPOINT
# done


# for NUM_TRAIN_POINTS in 50000 25000;
# do
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easymedium --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium$NUM_TRAIN_POINTS/ --seed 0
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
# done

# for NUM_TRAIN_POINTS in 37500 12500 6250;
# do
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easymedium --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium$NUM_TRAIN_POINTS/ --seed 0
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand$NUM_TRAIN_POINTS/ --seed 0
#    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS
#    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard$NUM_TRAIN_POINTS/ --seed 0
# done



# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points 6250
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard6250/ --seed 0



# torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 0 --eval_type train_part1
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 0 --eval_type train_part2


# for NUM_TRAIN_POINTS in 500 1000 2000;
# do
#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type easy5 --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_easy5$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_easy5$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
# done
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 3 --eval_type train_part1
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 3 --eval_type train_part2
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 4 --eval_type train_part1
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 4 --eval_type train_part2



CHECKPOINT=1000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type hard_subsample

CHECKPOINT=2000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type hard_subsample

CHECKPOINT=3000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type hard_subsample

CHECKPOINT=4000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type hard_subsample


CHECKPOINT=1000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type easymedium_subsample
CHECKPOINT=2000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type easymedium_subsample
CHECKPOINT=3000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type easymedium_subsample
CHECKPOINT=4000
ray stop
python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000_saveckpts/checkpoint-$CHECKPOINT/ --seed 0 --eval_type easymedium_subsample

