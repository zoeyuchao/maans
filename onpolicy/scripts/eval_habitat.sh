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
    CUDA_VISIBLE_DEVICES=0,1 python eval/eval_habitat.py --eval_episodes 1 --use_eval --split "train" --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --use_wandb --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 2 --episode_length 40 --max_episode_length 300 --num_env_steps 20000000 --ppo_epoch 10 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt"  --model_dir "/home/gaojiaxuan/onpolicy/onpolicy/envs/habitat/model/pretrained_models" --use_same_scene --scene_id 43 --use_delta_reward --use_different_start_pos --use_merge_partial_reward \
    --use_time_penalty --wandb_name "mapping" --user_name "gaojiaxuan" \
    --use_max --use_cnn_agent_id --use_new_trace --use_weight_trace --use_own \
    --grid_goal --use_grid_simple --use_goal --use_overlap_penalty --use_complete_reward \
    --use_single --grid_pos --grid_agent_id --cnn_use_attn --use_one_cnn_model --cuda --use_wandb --use_render --save_gifs
    echo "training is done!"
done
