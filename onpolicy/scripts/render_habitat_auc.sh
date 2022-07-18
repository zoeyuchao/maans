#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=3
algo="mappo"
exp="(seed1_final_best)mappo_3agents_distill_rl_eval26_auc"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3,2 python eval/eval_habitat_auc.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --split "train" --use_same_scene --scene_id 26 --eval_episodes 100 --use_eval --ifi 0.01 --seed 2 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 240 --num_local_steps 15 --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --model_dir "./results/Habitat/mappo/3agents_distill_best" --use_centralized_V --use_time_penalty --use_delta_reward --wandb_name "mapping" --use_max --use_cnn_agent_id --use_new_trace --use_weight_trace --use_own --grid_goal --use_grid_simple --use_goal --use_overlap_penalty --use_complete_reward --use_single --grid_pos --grid_agent_id --cnn_use_attn --use_different_start_pos --use_fixed_start_pos
    echo "training is done!" 
done