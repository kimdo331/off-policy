#!/bin/sh
env="StarCraft2"
map="3m"
algo="qmix"
exp="debug"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --buffer_size 5000 --lr 5e-4 --batch_size 32 --use_soft_update --hard_update_interval_episode 200 --num_env_steps 10000000 --log_interval 20000 --user_name "zoeyuchao"
    echo "training is done!"
done
