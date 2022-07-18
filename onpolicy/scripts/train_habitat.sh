#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1 python train/train_habitat.py --scenario_name ${scenario} \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} \
    --split "train" --seed 1 --n_training_threads 2 --n_rollout_threads 2 --num_mini_batch 5 \
    --num_local_steps 15 --max_episode_length 300 --num_env_steps 3000000 --ppo_epoch 4 --gain 0.01 \
    --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1' --hidden_size 256 --log_interval 1 \
    --use_recurrent_policy  \
    --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 10 \
    --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --use_centralized_V \
    --eval_episodes 1 --use_same_scene --scene_id 16 \
    --use_delta_reward --use_merge_partial_reward --use_time_penalty \
    --use_overlap_penalty --use_complete_reward --use_single \
    --wandb_name "wandb_name" --user_name "user_name" \
    --use_max --use_new_trace --use_weight_trace --use_goal \
    --use_own --grid_goal --use_different_start_pos --use_grid_simple \
    --grid_pos --grid_last_goal --cnn_use_transformer --use_share_cnn_model \
    --agent_invariant --invariant_type alter_attn --use_pos_embedding --use_id_embedding --multi_layer_cross_attn --add_grid_pos --use_self_attn --use_intra_attn --use_local_single_map \
    --use_wandb
    echo "training is done!" 
done