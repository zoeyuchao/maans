#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=3
algo="ft_wma_rrt" # ft_rrt ft_utility ft_apf  
exp="3agents_eval21"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python eval/eval_habitat_ft_auc.py --scenario_name ${scenario} \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} \
    --split "train" --use_same_scene --scene_id 21 --eval_episodes 10 --use_eval --ifi 0.01 --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 900 \
    --num_local_steps 15 --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 \
    --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params "32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1" \
    --hidden_size 256 --log_interval 1 --use_recurrent_policy \
    --load_slam ../envs/habitat/model/pretrained_models/slam_best.pt \
    --load_local ../envs/habitat/model/pretrained_models/local_best.pt --use_complete_reward --use_centralized_V \
    --use_delta_reward --wandb_name mapping --user_name yang-xy20 --use_max --use_max_map --use_different_start_pos \
    --use_fixed_start_pos --ft_use_random
    echo "evaluation is done!" 
done
