#!/bin/bash
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_full/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_rand_traincorrect1+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_subsample_traincorrect1+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_rand_traincorrect3+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_subsample_traincorrect3+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_rand_traincorrect5+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_subsample_traincorrect5+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_rand_traincorrect2+/ --seed 0
# python eval_math_with_gsm8k.py --ckpt_dir ckpts/math_fft_subsample_traincorrect2+/ --seed 0


# torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 0
# # python eval.py --ckpt_dir ckpts/gsm8k_fft_rand0+/ --seed 0
# # torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type easy --subsample_numtraincorrect 1
# # python eval.py --ckpt_dir ckpts/gsm8k_fft_easy1+/ --seed 0
# # torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 1
# python eval.py --ckpt_dir ckpts/gsm8k_fft_rand1+/ --seed 0
# # torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type easy --subsample_numtraincorrect 5
# python eval.py --ckpt_dir ckpts/gsm8k_fft_easy5+/ --seed 0
# # torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 5
# python eval.py --ckpt_dir ckpts/gsm8k_fft_rand5+/ --seed 0
# # torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type easy --subsample_numtraincorrect 20
# python eval.py --ckpt_dir ckpts/gsm8k_fft_easy20+/ --seed 0
# # torchrun --nproc_per_node=2 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 20
# python eval.py --ckpt_dir ckpts/gsm8k_fft_rand20+/ --seed 0


# python eval.py --ckpt_dir ckpts/gsm8k_fft_subsampled_synthetic/ --seed 0
# python eval.py --ckpt_dir ckpts/gsm8k_fft_subsampled_ground_truth/ --seed 0



# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-700/ --seed 0 --eval_type test
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-600/ --seed 0 --eval_type test
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-100/ --seed 0 --eval_type test
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-200/ --seed 0 --eval_type test
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-300/ --seed 0 --eval_type test
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-400/ --seed 0 --eval_type test
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-500/ --seed 0 --eval_type test

# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-100/ --seed 0 --eval_type train1000
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-200/ --seed 0 --eval_type train1000
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-300/ --seed 0 --eval_type train1000
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-400/ --seed 0 --eval_type train1000
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-500/ --seed 0 --eval_type train1000
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-600/ --seed 0 --eval_type train1000
# python eval_math.py --ckpt_dir ckpts/math_fft_full_save_5epochs/checkpoint-700/ --seed 0 --eval_type train1000


# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type easy --subsample_numtraincorrect 20
# python eval.py --ckpt_dir ckpts/gsm8k_fft_easy20+/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 20
# python eval.py --ckpt_dir ckpts/gsm8k_fft_rand20+/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type hard --subsample_numtraincorrect 20
# python eval.py --ckpt_dir ckpts/gsm8k_fft_hard20+/ --seed 0

# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type easy --subsample_numtraincorrect 40
# python eval.py --ckpt_dir ckpts/gsm8k_fft_easy40+/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 40
# python eval.py --ckpt_dir ckpts/gsm8k_fft_rand40+/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type hard --subsample_numtraincorrect 40
# python eval.py --ckpt_dir ckpts/gsm8k_fft_hard40+/ --seed 0

# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type easy --subsample_numtraincorrect 60
# python eval.py --ckpt_dir ckpts/gsm8k_fft_easy60+/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type rand --subsample_numtraincorrect 60
# python eval.py --ckpt_dir ckpts/gsm8k_fft_rand60+/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train.py --subsample_type hard --subsample_numtraincorrect 60
# python eval.py --ckpt_dir ckpts/gsm8k_fft_hard60+/ --seed 0


# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points 25000
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard25000/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points 25000
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand25000/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points 25000
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy25000/ --seed 0

# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points 50000
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard50000/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points 50000
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand50000/ --seed 0
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points 50000
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy50000/ --seed 0


# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easymedium --num_train_points 1562
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium1562/ --seed 0
# find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easymedium1562 -maxdepth 2 -type f -name "*.safetensors" -delete
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type medium --num_train_points 1562
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_medium1562/ --seed 0
# find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_medium1562 -maxdepth 2 -type f -name "*.safetensors" -delete
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points 1562
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy1562/ --seed 0
# find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easy1562 -maxdepth 2 -type f -name "*.safetensors" -delete
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points 1562
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand1562/ --seed 0
# find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_rand1562 -maxdepth 2 -type f -name "*.safetensors" -delete
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points 1562
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard1562/ --seed 0
# find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_hard1562 -maxdepth 2 -type f -name "*.safetensors" -delete


# for NUM_TRAIN_POINTS in 9375;
# do
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_medium$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easy$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_rand$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_hard$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
# done

# for NUM_TRAIN_POINTS in 18750;
# do
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easymedium --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easymedium$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
    
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_medium$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
    
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easy$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
    
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_rand$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
    
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_hard$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
# done

# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 1 --eval_type train_part1
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 1 --eval_type train_part2
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 2 --eval_type train_part1
# ray stop
# python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_full/ --seed 2 --eval_type train_part2



# for NUM_TRAIN_POINTS in 500 2000 4000;
# do
#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
# done

# for NUM_TRAIN_POINTS in 500 1000 2000 4000;
# do
#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_hard$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_hard$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
# done

# NUM_TRAIN_POINTS=50000
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard${NUM_TRAIN_POINTS}_saveckpts/ --seed 0


# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_saveckpts/ --seed 0


# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard50000_saveckpts/checkpoint-1000/ --seed 0 --eval_type hard_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard50000_saveckpts/checkpoint-1000/ --seed 0 --eval_type hard_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard50000_saveckpts/checkpoint-2000/ --seed 0 --eval_type hard_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard50000_saveckpts/checkpoint-3000/ --seed 0 --eval_type hard_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard50000_saveckpts/checkpoint-4000/ --seed 0 --eval_type hard_subsample







# NUM_TRAIN_POINTS=5000

# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_mixed$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete



# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type easy_full --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_easy_full$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_easy_full$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete


# torchrun --nproc_per_node=4 --master_port=1235 train_arithmetic.py --data_type hard_skip --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_arithmetic.py --ckpt_dir ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS
# find /data/katie_kang/stanford_alpaca/ckpts/arit_fft_hard_skip$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# for NUM_TRAIN_POINTS in 10000;
# do

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_medium$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete
# done

# for NUM_TRAIN_POINTS in 2000 4000 6000 8000;
# do

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_medium$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete


#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_hard$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_hard$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# done



# for NUM_TRAIN_POINTS in 15000 20000;
# do

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type medium --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_medium$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_medium$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# done


# for NUM_TRAIN_POINTS in 500 1000;
# do
#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_easy$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_gsm8k_aug.py --data_type mixed --num_train_points $NUM_TRAIN_POINTS
#     ray stop
#     python eval_gsm8k_aug.py --ckpt_dir ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS/ --seed 0
#     find /data/katie_kang/stanford_alpaca/ckpts/gsm8k_aug_fft_mixed$NUM_TRAIN_POINTS -maxdepth 2 -type f -name "*.safetensors" -delete

# done



# NUM_TRAIN_POINTS=10000
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_5epochs/ --seed 0 --eval_type train_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_5epochs/ --seed 0 --eval_type test
# # find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_5epochs -maxdepth 2 -type f -name "*.safetensors" -delete


# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy${NUM_TRAIN_POINTS}_5epochs/ --seed 0 --eval_type train_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy${NUM_TRAIN_POINTS}_5epochs/ --seed 0 --eval_type test
# # find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easy${NUM_TRAIN_POINTS}_5epochs -maxdepth 2 -type f -name "*.safetensors" -delete


# NUM_TRAIN_POINTS=10000
# torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easymedium --num_train_points $NUM_TRAIN_POINTS --num_epochs 5
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium${NUM_TRAIN_POINTS}_5epochs/ --seed 0 --eval_type train_subsample
# ray stop
# python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium${NUM_TRAIN_POINTS}_5epochs/ --seed 0 --eval_type test
# find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easymedium${NUM_TRAIN_POINTS}_5epochs -maxdepth 2 -type f -name "*.safetensors" -delete


# for NUM_EPOCHS in 3 4 6
# do 
#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type rand --num_train_points $NUM_TRAIN_POINTS --num_epochs $NUM_EPOCHS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type train_subsample
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type test
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_rand${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easy --num_train_points $NUM_TRAIN_POINTS --num_epochs $NUM_EPOCHS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type train_subsample
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easy${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type test
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easy${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs -maxdepth 2 -type f -name "*.safetensors" -delete

#     torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type easymedium --num_train_points $NUM_TRAIN_POINTS --num_epochs $NUM_EPOCHS
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type train_subsample
#     ray stop
#     python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_easymedium${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type test
#     find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_easymedium${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs -maxdepth 2 -type f -name "*.safetensors" -delete
# done


NUM_TRAIN_POINTS=10000

for NUM_EPOCHS in 6 10
do 
    torchrun --nproc_per_node=4 --master_port=1235 train_math_aug.py --data_type hard --num_train_points $NUM_TRAIN_POINTS --num_epochs $NUM_EPOCHS
    ray stop
    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type train_subsample
    ray stop
    python eval_math_aug.py --ckpt_dir ckpts/math_aug_fft_hard${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs/ --seed 0 --eval_type test
    find /data/katie_kang/stanford_alpaca/ckpts/math_aug_fft_hard${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs -maxdepth 2 -type f -name "*.safetensors" -delete
done