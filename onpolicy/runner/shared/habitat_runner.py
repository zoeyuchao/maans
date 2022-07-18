import time
import wandb
import os
import gym
import numpy as np
import imageio
import math
from collections import defaultdict, deque
from itertools import chain
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import pyastar2d
import torch
import torch.nn as nn
from torch.nn import functional as F
from onpolicy.envs.habitat.utils import pose as pu
import copy
import json
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.envs.habitat.model.model import Neural_SLAM_Module, Local_IL_Policy
from onpolicy.envs.habitat.utils.memory import FIFOMemory
from onpolicy.envs.habitat.utils.frontier import get_closest_frontier, get_frontier, nearest_frontier, max_utility_frontier, bfs_distance, rrt_global_plan, l2distance, voronoi_based_planning, WMA_RRT
from onpolicy.algorithms.utils.util import init, check
from onpolicy.envs.habitat.utils.direction_goal import goal_planner
from onpolicy.utils.apf import APF 
from icecream import ic

def _t2n(x):
    return x.detach().cpu().numpy()

def get_folders(dir, folders):
    get_dir = os.listdir(dir)
    for i in get_dir:          
        sub_dir = os.path.join(dir, i)
        if os.path.isdir(sub_dir): 
            folders.append(sub_dir) 
            get_folders(sub_dir, folders)

class HabitatRunner(Runner):
    def __init__(self, config):
        super(HabitatRunner, self).__init__(config)
        # init parameters
        self.init_hyper_parameters()
        # init keys
        self.init_keys()
        # init variables
        self.init_map_variables() 
        # global policy
        self.init_global_policy() 
        # local policy
        self.init_local_policy()  
        # slam module
        self.init_slam_module()    
    
    def warmup(self):
        # reset env
        self.obs, infos = self.envs.reset()

        self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
        self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
        self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
        self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
        self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
        self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
        self.sim_map_size = [infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)]
        self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
        
        for agent_id in range(self.num_agents):
            self.intrinsic_gt[ : , agent_id] = self.explorable_map[ : , agent_id]

        # Predict map from frame 1:
        self.run_slam_module(self.obs, self.obs, infos)

        # Compute Global policy input        
        self.first_compute_global_input()
        if not self.use_centralized_V:
            self.share_global_input = self.global_input.copy()

        # replay buffer
        for key in self.global_input.keys():
            self.buffer.obs[key][0] = self.global_input[key].copy()

        for key in self.share_global_input.keys():
            self.buffer.share_obs[key][0] = self.share_global_input[key].copy()

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step=0)

        # compute local input
        if self.use_local_single_map:
            self.compute_local_input(self.single_merge_map)
        else:
            self.compute_local_input(self.merge_map)

        # Output stores local goals as well as the the ground-truth action
        self.global_output = self.envs.get_short_term_goal(self.global_insert)
        self.global_output = np.array(self.global_output, dtype = np.long)
        
        self.last_obs = copy.deepcopy(self.obs)
            
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
 
    def run(self):
        # map and pose
        self.init_map_and_pose()
        self.env_step = 0

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.max_episode_length // self.n_rollout_threads

        for episode in range(episodes):
            self.init_env_info()
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
    
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            auc_area = np.zeros((self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)
            for step in range(self.max_episode_length):
                # if (step+1) % 15  == 0:
                #     print("step", step+1)
                self.env_step = step
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                del self.last_obs
                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)
                self.rewards += reward
                for e in range (self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio':
                                auc_area[e, step] = auc_area[e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio':
                                auc_single_area[e, :, step] = auc_single_area[e, :, step-1] + np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'explored_ratio_step':
                            for agent_id in range(self.num_agents):
                                agent_k = "agent{}_{}".format(agent_id, key)
                                if agent_k in infos[e].keys():
                                    self.env_info[key][e][agent_id] = infos[e][agent_k]
                        else:
                            if key in infos[e].keys():
                                self.env_info[key][e] = infos[e][key]
                    if self.num_agents==1:
                        if step in [49, 99, 149, 199, 249, 299, 349, 399, 449,359,719]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio']
                    else:
                        if step in [49, 99, 149, 199, 249, 299, 349, 399, 449,359,719]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio'] 
                              
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                # Reinitialize variables when episode ends
                if step == self.max_episode_length - 1:
                    self.obs, infos = self.envs.reset()
                    self.init_map_and_pose()
                    del self.last_obs
                    self.last_obs = copy.deepcopy(self.obs)
                    self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
                    self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
                    self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
                    self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
                    self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
                    self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
                    self.sim_map_size = [infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)]
                    self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
                    if self.action_mask:
                        self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                        self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                   
                    for agent_id in range(self.num_agents):
                        self.intrinsic_gt[:, agent_id] = self.explorable_map[:, agent_id]

                # Neural SLAM Module
                if self.train_slam:
                    self.insert_slam_module(infos)
                
                self.run_slam_module(self.last_obs, self.obs, infos, True)
                self.update_local_map()
                self.update_map_and_pose(False)

                for a in range(self.num_agents):
                    if self.use_local_single_map:
                        self.single_merge_map[:, a] = self.single_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    else:
                        self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)   
                
                # Global Policy
                if local_step == self.num_local_steps - 1:
                    # For every global step, update the full and local maps
                    self.update_map_and_pose()
                    self.compute_global_input()
                    data = dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                    # insert data into buffer
                    self.insert_global_policy(data)
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step = global_step + 1)

                # Local Policy
                if self.use_local_single_map:
                    self.compute_local_input(self.single_merge_map)
                else:
                    self.compute_local_input(self.merge_map)

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.global_insert)
                self.global_output = np.array(self.global_output, dtype = np.long)

                # Start Training
                torch.set_grad_enabled(True)

                # Train Neural SLAM Module
                if self.train_slam and len(self.slam_memory) > self.slam_batch_size:
                    self.train_slam_module()
                    
                # Train Local Policy
                if self.train_local and (local_step + 1) % self.local_policy_update_freq == 0:
                    self.train_local_policy()
                    
                # Train Global Policy
                if global_step % self.episode_length == self.episode_length - 1 \
                        and local_step == self.num_local_steps - 1:
                    self.train_global_policy()
                    
                # Finish Training
                torch.set_grad_enabled(False)
                
            # post process
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            
            self.convert_info()
            print("average episode merge explored reward is {}".format(np.mean(self.env_infos["sum_merge_explored_reward"])))
            print("average episode merge explored ratio is {}".format(np.mean(self.env_infos['sum_merge_explored_ratio'])))
            print("average episode merge repeat area is {}".format(np.mean(self.env_infos['sum_merge_repeat_area'])))

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_env(self.train_slam_infos, total_num_steps)
                self.log_env(self.train_local_infos, total_num_steps)
                self.log_env(self.train_global_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.log_single_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
            #save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save_slam_model(total_num_steps)
                self.save_global_model(total_num_steps)
                self.save_local_model(total_num_steps)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_hyper_parameters(self):
        self.map_size_cm = self.all_args.map_size_cm
        self.map_resolution = self.all_args.map_resolution
        self.global_downscaling = self.all_args.global_downscaling

        self.frame_width = self.all_args.frame_width
       
        self.load_local = self.all_args.load_local
        self.load_slam = self.all_args.load_slam
        self.train_local = self.all_args.train_local
        self.train_slam = self.all_args.train_slam

        self.proj_frontier = self.all_args.proj_frontier

        self.slam_memory_size = self.all_args.slam_memory_size
        self.slam_batch_size = self.all_args.slam_batch_size
        self.slam_iterations = self.all_args.slam_iterations
        self.slam_lr = self.all_args.slam_lr
        self.slam_opti_eps = self.all_args.slam_opti_eps

        self.use_local_recurrent_policy = self.all_args.use_local_recurrent_policy
        self.local_hidden_size = self.all_args.local_hidden_size
        self.local_lr = self.all_args.local_lr
        self.local_opti_eps = self.all_args.local_opti_eps

        self.proj_loss_coeff = self.all_args.proj_loss_coeff
        self.exp_loss_coeff = self.all_args.exp_loss_coeff
        self.pose_loss_coeff = self.all_args.pose_loss_coeff

        self.local_policy_update_freq = self.all_args.local_policy_update_freq
        self.num_local_steps = self.all_args.num_local_steps
        self.max_episode_length = self.all_args.max_episode_length
        self.render_merge = self.all_args.render_merge
        self.visualize_input = self.all_args.visualize_input
        self.use_intrinsic_reward = self.all_args.use_intrinsic_reward
        self.use_delta_reward = self.all_args.use_delta_reward
        self.use_abs_orientation = self.all_args.use_abs_orientation
        
        self.use_resnet = self.all_args.use_resnet
        self.map_threshold = self.all_args.map_threshold
        self.use_max = self.all_args.use_max
        self.use_orientation = self.all_args.use_orientation
        self.use_vector_agent_id = self.all_args.use_vector_agent_id
        self.use_cnn_agent_id = self.all_args.use_cnn_agent_id
        self.use_own = self.all_args.use_own
        self.use_new_trace = self.all_args.use_new_trace
        self.use_weight_trace = self.all_args.use_weight_trace
        self.decay_weight = self.all_args.decay_weight
        self.use_merge_goal = self.all_args.use_merge_goal
        self.use_single_agent_trace = self.all_args.use_single_agent_trace
        self.use_local = self.all_args.use_local
        self.local_map_w = self.all_args.local_map_w
        self.local_map_h = self.all_args.local_map_h
        self.use_seperated_cnn_model = self.all_args.use_seperated_cnn_model
        self.grid_pos = self.all_args.grid_pos
        self.grid_agent_id = self.all_args.grid_agent_id
        self.agent_invariant = self.all_args.agent_invariant

        self.grid_size = self.all_args.grid_size
        self.discrete_goal = self.all_args.discrete_goal
        self.multidiscrete_goal = self.all_args.multidiscrete_goal
        self.direction_goal = self.all_args.direction_goal
        self.grid_goal = self.all_args.grid_goal
        self.action_mask = self.all_args.action_mask
        self.grid_goal_simpler = self.all_args.grid_goal_simpler
        self.use_goal_penalty = self.all_args.use_goal_penalty
        self.use_pos_penalty = self.all_args.use_pos_penalty
        self.use_single = self.all_args.use_single
        self.use_map_agent_id = self.all_args.use_map_agent_id
        self.use_goal = self.all_args.use_goal
        self.use_single_merge = self.all_args.use_single_merge
        self.use_local_single_map = self.all_args.use_local_single_map
        self.use_agent_id = self.all_args.use_agent_id
        self.use_self_grid_pos = self.all_args.use_self_grid_pos
        self.grid_last_goal = self.all_args.grid_last_goal
        
    def init_map_variables(self):
        ### Full map consists of 4 channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations

        # Calculating full and local map sizes
        map_size = self.map_size_cm // self.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w, self.local_h = int(self.full_w / self.global_downscaling), \
                        int(self.full_h / self.global_downscaling)
        self.center_w, self.center_h = self.full_w//2, self.full_h//2
        if self.use_resnet:
            self.res_w = self.res_h = 224

        # Initializing full, merge and local map
        self.full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.local_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h), dtype=np.float32)
        
        # Initial full and local pose
        self.full_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)
        self.local_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)

        # Origin of local map
        self.origins = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)

        # Local Map Boundaries
        self.lmb = np.zeros((self.n_rollout_threads, self.num_agents, 4)).astype(int)
        
        self.local_lmb = np.zeros((self.n_rollout_threads, self.num_agents, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.n_rollout_threads, self.num_agents, 7), dtype=np.float32)
        
        # each agent rotation
        self.other_agent_rotation = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        if self.use_local:
            self.agent_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.full_w, self.full_h), dtype=np.float32)
        if self.use_cnn_agent_id:
            self.agent_pos_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_single_agent_trace:
            self.agent_trace_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.grid_agent_id or self.grid_pos:
            self.grid_locs = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        self.world_locs = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        if self.grid_last_goal:
            self.grid_goal_pos = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)

        # ft
        self.ft_merge_map = np.zeros((self.n_rollout_threads, 2, self.full_w, self.full_h), dtype = np.float32) # only explored and obstacle
        self.ft_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype = np.int32)
        self.ft_pre_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype = np.int32)
        self.ft_last_merge_explored_ratio = np.zeros((self.n_rollout_threads, 1), dtype= np.float32)
        self.ft_mask = np.ones((self.full_w, self.full_h), dtype=np.int32)
        self.ft_go_steps = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype= np.int32)
        self.ft_map = [None for _ in range(self.n_rollout_threads)]
        self.ft_lx = [None for _ in range(self.n_rollout_threads)]
        self.ft_ly = [None for _ in range(self.n_rollout_threads)]
               
    def init_map_and_pose(self):
        self.full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.full_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)
        self.intrinsic_gt = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.merge_goal_trace = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.full_pose[:, :, :2] = self.map_size_cm / 100.0 / 2.0

        locs = self.full_pose
        self.planner_pose_inputs[:, :, :3] = locs
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.full_map[e, a, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1

                self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                    (self.local_w, self.local_h),
                                                    (self.full_w, self.full_h))

                self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a].copy()
                self.origins[e, a] = [self.lmb[e, a, 2] * self.map_resolution / 100.0,
                                self.lmb[e, a, 0] * self.map_resolution / 100.0, 0.]

        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
                self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]

    def init_keys(self):
        # train keys
        self.train_global_infos_keys = ['value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','ratio','std_advantages']
        self.train_local_infos_keys = ['local_policy_loss']
        self.train_slam_infos_keys = ['costs','exp_costs','pose_costs']

        # info keys
        self.sum_env_info_keys = ['overlap_reward','explored_ratio', 'merge_explored_ratio', 'merge_explored_reward', 'explored_reward', 'repeat_area', 'merge_repeat_area']
        self.equal_env_info_keys = ['merge_overlap_ratio',  'merge_overlap_ratio_0.3', 'merge_overlap_ratio_0.5', 'merge_overlap_ratio_0.7', 'merge_explored_ratio_step', 'merge_explored_ratio_step_0.95', 'explored_ratio_step']
        
        # log keys
        if self.num_agents==1:
            self.agents_env_info_keys = ['sum_explored_ratio','sum_overlap_reward','sum_explored_reward','sum_intrinsic_merge_explored_reward','sum_repeat_area','explored_ratio_step',\
                '50step_auc','100step_auc','150step_auc','200step_auc','250step_auc','300step_auc','350step_auc','400step_auc','450step_auc','360step_auc','720step_auc']
            self.env_info_keys = ['sum_merge_explored_ratio','sum_merge_explored_reward','sum_merge_repeat_area','merge_overlap_ratio', 'merge_overlap_ratio_0.3', 'merge_overlap_ratio_0.5', 'merge_overlap_ratio_0.7', 'merge_explored_ratio_step','merge_explored_ratio_step_0.95', 'merge_global_goal_num', 'merge_global_goal_num_%.2f'%self.all_args.explored_ratio_down_threshold,\
                '50step_merge_auc','100step_merge_auc','150step_merge_auc','200step_merge_auc','250step_merge_auc','300step_merge_auc','350step_merge_auc','400step_merge_auc','450step_merge_auc',
                '50step_merge_overlap_ratio','100step_merge_overlap_ratio','150step_merge_overlap_ratio','200step_merge_overlap_ratio','250step_merge_overlap_ratio','300step_merge_overlap_ratio','350step_merge_overlap_ratio','400step_merge_overlap_ratio','450step_merge_overlap_ratio','360step_merge_auc','720step_merge_auc','360step_merge_overlap_ratio','720step_merge_overlap_ratio']
        else:
            self.agents_env_info_keys = ['sum_explored_ratio','sum_overlap_reward','sum_explored_reward','sum_intrinsic_merge_explored_reward','sum_repeat_area','explored_ratio_step',\
                '50step_auc','100step_auc','120step_auc','150step_auc','180step_auc','200step_auc','250step_auc','300step_auc','210step_auc','90step_auc','600step_auc','480step_auc','420step_auc']
            self.env_info_keys = ['sum_merge_explored_ratio','sum_merge_explored_reward','sum_merge_repeat_area','merge_overlap_ratio', 'merge_overlap_ratio_0.3', 'merge_overlap_ratio_0.5', 'merge_overlap_ratio_0.7', 'merge_explored_ratio_step','merge_explored_ratio_step_0.95', 'merge_global_goal_num', 'merge_global_goal_num_%.2f'%self.all_args.explored_ratio_down_threshold,\
                '50step_merge_auc','100step_merge_auc','120step_merge_auc','150step_merge_auc','180step_merge_auc','200step_merge_auc','250step_merge_auc','300step_merge_auc',
                '50step_merge_overlap_ratio','100step_merge_overlap_ratio','120step_merge_overlap_ratio','150step_merge_overlap_ratio','180step_merge_overlap_ratio','200step_merge_overlap_ratio','250step_merge_overlap_ratio','300step_merge_overlap_ratio',\
                '210step_merge_overlap_ratio','90step_merge_overlap_ratio','600step_merge_overlap_ratio','480step_merge_overlap_ratio','420step_merge_overlap_ratio',\
                '210step_merge_auc','90step_merge_auc','600step_merge_auc','480step_merge_auc','420step_merge_auc']
                
        if self.use_eval:
            self.agents_env_info_keys += ['sum_path_length', 'path_length/ratio', "balanced_ratio"]
            self.sum_env_info_keys  += ['path_length']
            self.equal_env_info_keys  += ['path_length/ratio',"balanced_ratio"]
            self.auc_infos_keys = ['merge_auc','agent_auc']
            self.env_info_keys += ['merge_runtime'] 

        # convert keys
        self.env_infos_keys = self.agents_env_info_keys + self.env_info_keys + \
                        ['max_sum_merge_explored_ratio','min_sum_merge_explored_ratio','merge_success_rate','invalid_merge_explored_ratio_step_num','invalid_merge_map_num'] 

    def init_global_policy(self):
        self.best_gobal_reward = -np.inf
        length = 1
        # ppo network log info
        self.train_global_infos = {}
        for key in self.train_global_infos_keys:
            self.train_global_infos[key] = deque(maxlen=length)

        # env info
        self.env_infos = {}
        for key in self.env_infos_keys:
            self.env_infos[key] = deque(maxlen=length)

        # auc info
        if self.use_eval:
            self.auc_infos = {}
            for key in self.auc_infos_keys:
                if 'merge' in key:
                    self.auc_infos[key] = np.zeros((self.all_args.eval_episodes, self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
                else:
                    self.auc_infos[key] = np.zeros((self.all_args.eval_episodes,  self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)

        if self.direction_goal:
            self.goal_planner = goal_planner(self.all_args, self.run_dir)

        self.global_input = {}
        
            
        if self.action_mask:
            self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
            self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

        if self.grid_agent_id:
            self.global_input['grid_agent_id'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
        if self.grid_pos:
            if self.use_self_grid_pos:
                self.global_input['grid_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.grid_size, self.grid_size), dtype=np.int32)
            else:
                self.global_input['grid_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents*2, self.grid_size, self.grid_size), dtype=np.int32)
        
        if self.grid_last_goal:
            if self.use_self_grid_pos:
                self.global_input['grid_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.grid_size, self.grid_size), dtype=np.int32)
            else:
                self.global_input['grid_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents*2, self.grid_size, self.grid_size), dtype=np.int32)

        if self.use_resnet:
            if self.use_single:
                self.global_input['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.res_w, self.res_h), dtype=np.float32)
                if self.use_goal:
                    self.global_input['global_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.res_w, self.res_h), dtype=np.float32)
                if self.use_single_merge:
                    self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.res_w, self.res_h), dtype=np.float32)
            else:
                if self.use_local:
                    if self.use_seperated_cnn_model:
                        self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.res_w, self.res_h), dtype=np.float32)
                        self.global_input['local_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.res_w, self.res_h), dtype=np.float32)
                    else:
                        self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.res_w, self.res_h), dtype=np.float32)
                else:
                    self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.res_w, self.res_h), dtype=np.float32)
            #self.global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.res_w, self.res_h), dtype=np.float32)
            if self.use_single_agent_trace:
                if self.use_own:
                    self.global_input['trace_image'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.res_w, self.res_h), dtype=np.float32)
                else:
                    self.global_input['trace_image'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, self.res_w, self.res_h), dtype=np.float32)
            if self.use_merge_goal:
                self.global_input['global_merge_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.res_w, self.res_h), dtype=np.float32)
        else:
            if self.use_single:
                self.global_input['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h), dtype=np.float32)
                if self.use_goal:
                    self.global_input['global_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_w, self.local_h), dtype=np.float32)
                if self.use_single_merge:
                    self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_w, self.local_h), dtype=np.float32)
            else:
                if self.use_local:
                    if self.use_seperated_cnn_model:
                        self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_map_w, self.local_map_h), dtype=np.float32)
                        self.global_input['local_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_map_w, self.local_map_h), dtype=np.float32)
                    else:
                        self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.local_map_w, self.local_map_h), dtype=np.float32)
                    if self.use_merge_goal:
                        self.global_input['global_merge_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_map_w, self.local_map_h), dtype=np.float32)
                else:
                    self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h), dtype=np.float32)
                    
            
            #self.global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_w, self.local_h), dtype=np.float32)
            if self.use_single_agent_trace:
                if self.use_local:
                    if self.use_own:
                        self.global_input['trace_image'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_map_w, self.local_map_h), dtype=np.float32)
                    else:
                        self.global_input['trace_image'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, self.local_map_w, self.local_map_h), dtype=np.float32)
                        
                else:
                    if self.use_own:
                        self.global_input['trace_image'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_w, self.local_h), dtype=np.float32)
                    else:
                        self.global_input['trace_image'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, self.local_w, self.local_h), dtype=np.float32)          
            if self.use_merge_goal:
                self.global_input['global_merge_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_w, self.local_h), dtype=np.float32) 
        
        if self.use_orientation:
            self.global_input['global_orientation'] = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.long)
            self.global_input['other_global_orientation'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents-1), dtype=np.long)
        
        if self.use_vector_agent_id:
            self.global_input['vector'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
        
        if self.use_cnn_agent_id:
            if self.use_resnet:
                self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, 224, 224), dtype=np.float32)
                # if self.use_own:
                #     self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, 224, 224), dtype=np.float32)
                # else:
                #     self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents+1, 224, 224), dtype=np.float32)
            else:
                if self.use_local:
                    self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_map_w, self.local_map_h), dtype=np.float32)
                    # if self.use_own:
                    #     self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_map_w, self.local_map_h), dtype=np.float32)
                    # else:
                    #     self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents+1, self.local_map_w, self.local_map_h), dtype=np.float32)
                else:
                    self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_w, self.local_h), dtype=np.float32)
                    # if self.use_own:
                    #     self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_w, self.local_h), dtype=np.float32)
                    # else:
                    #     self.global_input['vector_cnn'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents+1, self.local_w, self.local_h), dtype=np.float32)

        self.share_global_input = self.global_input.copy()
        
        if self.use_centralized_V:
            if self.use_resnet:
                self.share_global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.res_w, self.res_h), dtype=np.float32)
            else:
                if self.use_local:
                    self.share_global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_map_w, self.local_map_h), dtype=np.float32)
                else:
                    self.share_global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_w, self.local_h), dtype=np.float32)
        
        if self.direction_goal:
            if self.use_resnet:
                self.global_input['direction_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 3 * self.all_args.direction_k + 1, self.res_w, self.res_h), dtype=np.float32)
            else:
                self.global_input['direction_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 3 * self.all_args.direction_k + 1, self.local_w, self.local_h), dtype=np.float32)
            self.global_input['direction_vec'] = np.zeros((self.n_rollout_threads, self.num_agents, self.all_args.direction_k), dtype=np.float32)

        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
        self.revise_global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

        self.first_compute = True
        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(self.num_agents*2, 8, figsize=(10, 2.5), facecolor="whitesmoke")
            self.fig_d, self.ax_d = plt.subplots(self.num_agents, 3 * self.all_args.direction_k + 1, figsize=(10, 2.5), facecolor = "whitesmoke")

    def init_local_policy(self):
        self.best_local_loss = np.inf

        self.train_local_infos = {}
        for key in self.train_local_infos_keys:
            self.train_local_infos[key] = deque(maxlen=1000)
        
        # Local policy
        self.local_masks = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.local_rnn_states = torch.zeros((self.n_rollout_threads, self.num_agents, self.local_hidden_size)).to(self.device)
        
        local_observation_space = gym.spaces.Box(0, 255, (3,
                                                    self.frame_width,
                                                    self.frame_width), dtype='uint8')
        local_action_space = gym.spaces.Discrete(3)

        self.local_policy = Local_IL_Policy(local_observation_space.shape, local_action_space.n,
                               recurrent=self.use_local_recurrent_policy,
                               hidden_size=self.local_hidden_size,
                               deterministic=self.all_args.use_local_deterministic,
                               device=self.device)
        
        if self.load_local != "0":
            print("Loading local {}".format(self.load_local))
            state_dict = torch.load(self.load_local, map_location=self.device)
            self.local_policy.load_state_dict(state_dict)

        if not self.train_local:
            self.local_policy.eval()
        else:
            self.local_policy_loss = 0
            self.local_optimizer = torch.optim.Adam(self.local_policy.parameters(), lr=self.local_lr, eps=self.local_opti_eps)

    def init_slam_module(self):
        self.best_slam_cost = 10000
        
        self.train_slam_infos = {}
        for key in self.train_slam_infos_keys:
            self.train_slam_infos[key] = deque(maxlen=1000)
        
        self.nslam_module = Neural_SLAM_Module(self.all_args, device=self.device)
        
        if self.load_slam != "0":
            print("Loading slam {}".format(self.load_slam))
            state_dict = torch.load(self.load_slam, map_location=self.device)
            self.nslam_module.load_state_dict(state_dict)
        
        if not self.train_slam:
            self.nslam_module.eval()
        else:
            self.slam_memory = FIFOMemory(self.slam_memory_size)
            self.slam_optimizer = torch.optim.Adam(self.nslam_module.parameters(), lr=self.slam_lr, eps=self.slam_opti_eps)

    def init_env_info(self):
        self.env_info = {}

        for key in self.agents_env_info_keys:
            if "step" in key:
                self.env_info[key] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) * self.max_episode_length
            else:
                self.env_info[key] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        
        for key in self.env_info_keys:
            if "step" in key:
                self.env_info[key] = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.max_episode_length
            else:
                self.env_info[key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
                    
    def convert_info(self):
        for k, v in self.env_info.items():
            if k == "explored_ratio_step":
                self.env_infos[k].append(v)
                for agent_id in range(self.num_agents):
                    print("agent{}_{}: {}/{}".format(agent_id, k, np.mean(v[:, agent_id]), self.max_episode_length))
                print('minimal agent {}: {}/{}'.format(k, np.min(v), self.max_episode_length))
            elif k == "merge_explored_ratio_step":
                print('invaild {} map num: {}/{}'.format(k, (v == self.max_episode_length).sum(), self.n_rollout_threads))
                self.env_infos['invalid_merge_map_num'].append((v == self.max_episode_length).sum())
                self.env_infos['merge_success_rate'].append((v != self.max_episode_length).sum() / self.n_rollout_threads)
                if (v == self.max_episode_length).sum() > 0:
                    scene_id = np.argwhere(v == self.max_episode_length).reshape((v == self.max_episode_length).sum())
                    if self.all_args.use_same_scene:
                        print('invaild {} map id: {}'.format(k, self.scene_id[scene_id[0]]))
                    else:
                        for i in range(len(scene_id)):
                            print('invaild {} map id: {}'.format(k, self.scene_id[scene_id[i]]))
                v_copy = v.copy()
                v_copy[v == self.max_episode_length] = np.nan
                self.env_infos[k].append(v)
                print('mean valid {}: {}'.format(k, np.nanmean(v_copy)))
            else:
                if k != 'auc':
                    self.env_infos[k].append(v)
                if k == 'sum_merge_explored_ratio':       
                    self.env_infos['max_sum_merge_explored_ratio'].append(np.max(v))
                    self.env_infos['min_sum_merge_explored_ratio'].append(np.min(v))
                    print(np.mean(v))

    def insert_slam_module(self, infos):
        # Add frames to memory
        for agent_id in range(self.num_agents):
            for env_idx in range(self.n_rollout_threads):
                env_poses = infos[env_idx]['sensor_pose'][agent_id]
                env_gt_fp_projs = np.expand_dims(infos[env_idx]['fp_proj'][agent_id], 0)
                env_gt_fp_explored = np.expand_dims(infos[env_idx]['fp_explored'][agent_id], 0)
                env_gt_pose_err = infos[env_idx]['pose_err'][agent_id]
                self.slam_memory.push(
                    (self.last_obs[env_idx][agent_id], self.obs[env_idx][agent_id], env_poses),
                    (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

    def run_slam_module(self, last_obs, obs, infos, build_maps=True):
        for a in range(self.num_agents):
            poses = np.array([infos[e]['sensor_pose'][a] for e in range(self.n_rollout_threads)])

            _, _, self.local_map[:, a, 0, :, :], self.local_map[:, a, 1, :, :], _, self.local_pose[:, a, :] = \
                self.nslam_module(last_obs[:, a, :, :, :], obs[:, a, :, :, :], poses, 
                                self.local_map[:, a, 0, :, :],
                                self.local_map[:, a, 1, :, :], 
                                self.local_pose[:, a, :],
                                build_maps = build_maps)
    
    def transform(self, inputs, trans, rotation, agent_trans, agent_rotation, a):
        merge_map = np.zeros((self.n_rollout_threads, 4, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                output = torch.from_numpy(inputs[e, agent_id])  
                n_rotated = F.grid_sample(output.unsqueeze(0).float(), rotation[e][agent_id].float(), align_corners=True)
                n_map = F.grid_sample(n_rotated.float(), trans[e][agent_id].float(), align_corners=True)
                     
                agent_merge_map = n_map[0, :, :, :].numpy()

                (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[2, :, :], axis=None), agent_merge_map[2, :, :].shape)
                
                agent_merge_map[2, :, :] = np.zeros((self.full_h, self.full_w), dtype=np.float32)
                if self.first_compute:
                    if self.use_agent_id:
                        agent_merge_map[2, index_a - 1: index_a + 2, index_b - 1: index_b + 2] = (agent_id + 1)/np.array([aa + 1 for aa in range(self.num_agents)]).sum()
                    else:
                        agent_merge_map[2, index_a - 1: index_a + 2, index_b - 1: index_b + 2] = 1
                else: 
                    if self.use_agent_id:
                        agent_merge_map[2, index_a - 2: index_a + 3, index_b - 2: index_b + 3] = (agent_id + 1)/np.array([aa + 1 for aa in range(self.num_agents)]).sum()
                    else:
                        agent_merge_map[2, index_a - 2: index_a + 3, index_b - 2: index_b + 3] = 1
            
                trace = np.zeros((self.full_h, self.full_w), dtype=np.float32)
                #trace[0][agent_merge_map[0] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                #trace[1][agent_merge_map[1] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                if self.use_agent_id:
                    trace[agent_merge_map[3] > self.map_threshold] = (agent_id + 1)/np.array([aa+1 for aa in range(self.num_agents)]).sum()
                else:
                    trace[agent_merge_map[3] > self.map_threshold] = 1
                    
                #agent_merge_map[0:2] = trace[0:2]
                if self.use_new_trace and self.use_weight_trace:
                    agent_merge_map[3] *= trace
                else:
                    agent_merge_map[3] = trace
                if self.use_max:
                    for i in range(2):
                        merge_map[e, i] = np.maximum(merge_map[e, i], agent_merge_map[i])
                        merge_map[e, i+2] += agent_merge_map[i+2]
                else:
                    merge_map[e] += agent_merge_map
                if self.use_cnn_agent_id:    
                    self.agent_pos_map[e, agent_id] = agent_merge_map[2]
                if self.use_single_agent_trace:
                    self.agent_trace_map[e, agent_id] = agent_merge_map[3]
                if self.use_local:
                    self.agent_merge_map[e, agent_id] = agent_merge_map[2:]
                if self.grid_agent_id or self.grid_pos:
                    self.grid_locs[e, agent_id, 0] = int((index_a-(self.full_w/2-self.sim_map_size[e][agent_id][0]//2-5))/((self.sim_map_size[e][agent_id][0]+10)/self.grid_size))
                    self.grid_locs[e, agent_id, 1] = int((index_b-(self.full_h/2-self.sim_map_size[e][agent_id][1]//2-5))/((self.sim_map_size[e][agent_id][1]+10)/self.grid_size))
                self.world_locs[e, agent_id, 0] = index_a
                self.world_locs[e, agent_id, 1] = index_b
    
            # agent_n_trans = F.grid_sample(torch.from_numpy(merge_map[e]).unsqueeze(0).float(), agent_trans[e][a].float(), align_corners=True)      
            # merge_map[e] = F.grid_sample(agent_n_trans.float(), agent_rotation[e][a].float(), align_corners=True)[0, :, :, :].numpy()
            if not self.use_max:
                for i in range(2):
                    merge_map[ e, i][merge_map[ e, i]>1] = 1.0
                    merge_map[ e, i][merge_map[ e, i]<self.map_threshold] = 0.0

        return merge_map
    
    def single_transform(self, inputs, trans, rotation, agent_trans, agent_rotation, a):
        single_map = np.zeros((self.n_rollout_threads, 4, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            output = torch.from_numpy(inputs[e, a])  
            n_rotated = F.grid_sample(output.unsqueeze(0).float(), rotation[e][a].float(), align_corners=True)
            n_map = F.grid_sample(n_rotated.float(), trans[e][a].float(), align_corners=True)
                    
            agent_merge_map = n_map[0, :, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[2, :, :], axis=None), agent_merge_map[2, :, :].shape)
            
            agent_merge_map[2, :, :] = np.zeros((self.full_h, self.full_w), dtype=np.float32)
            if self.first_compute:
                if self.use_agent_id:
                    agent_merge_map[2, index_a - 1: index_a + 2, index_b - 1: index_b + 2] = (a + 1)/np.array([aa + 1 for aa in range(self.num_agents)]).sum()
                else:
                    agent_merge_map[2, index_a - 1: index_a + 2, index_b - 1: index_b + 2] = 1
            else: 
                if self.use_agent_id:
                    agent_merge_map[2, index_a - 2: index_a + 3, index_b - 2: index_b + 3] = (a + 1)/np.array([aa + 1 for aa in range(self.num_agents)]).sum()
                else:
                    agent_merge_map[2, index_a - 2: index_a + 3, index_b - 2: index_b + 3] = 1
        
            trace = np.zeros((self.full_h, self.full_w), dtype=np.float32)
            #trace[0][agent_merge_map[0] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
            #trace[1][agent_merge_map[1] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
            if self.use_agent_id:
                trace[agent_merge_map[3] > self.map_threshold] = (a + 1)/np.array([aa+1 for aa in range(self.num_agents)]).sum()
            else:
                trace[agent_merge_map[3] > self.map_threshold] = 1
            #agent_merge_map[0:2] = trace[0:2]
            if self.use_new_trace and self.use_weight_trace:
                agent_merge_map[3] *= trace
            else:
                agent_merge_map[3] = trace
            
            if self.use_cnn_agent_id:    
                self.agent_pos_map[e, a] = agent_merge_map[2]
            if self.use_single_agent_trace:
                self.agent_trace_map[e, a] = agent_merge_map[3]
            if self.use_local:
                self.agent_merge_map[e, a] = agent_merge_map[2:]
            if self.grid_agent_id or self.grid_pos:
                self.grid_locs[e, a, 0] = int((index_a-(self.full_w/2-self.sim_map_size[e][a][0]//2-5))/((self.sim_map_size[e][a][0]+10)/self.grid_size))
                self.grid_locs[e, a, 1] = int((index_b-(self.full_h/2-self.sim_map_size[e][a][1]//2-5))/((self.sim_map_size[e][a][1]+10)/self.grid_size))
            self.world_locs[e, a, 0] = index_a
            self.world_locs[e, a, 1] = index_b
    
            # agent_n_trans = F.grid_sample(torch.from_numpy(merge_map[e]).unsqueeze(0).float(), agent_trans[e][a].float(), align_corners=True)      
            # merge_map[e] = F.grid_sample(agent_n_trans.float(), agent_rotation[e][a].float(), align_corners=True)[0, :, :, :].numpy()
            single_map[e] = agent_merge_map
        return single_map
    
    def transform_gt(self, inputs, trans, rotation, agent_id):
        gt_world_map = np.zeros((self.n_rollout_threads, 1, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            output = torch.from_numpy(inputs[e])  
            n_rotated = F.grid_sample(output.unsqueeze(0).unsqueeze(0).float(), rotation[e][agent_id].float(), align_corners=True)
            n_map = F.grid_sample(n_rotated.float(), trans[e][agent_id].float(), align_corners=True)    
            gt_world_map[e] = n_map[0, :, :, :].numpy()

        return gt_world_map
    
    def center_transform(self, inputs, a):
        merge_map = np.zeros((self.n_rollout_threads, 4, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            for i in range(4):
                r, c = self.full_pose[e, a,:2]
                r, c =[int(r * 100.0 / self.map_resolution), int(c * 100.0 / self.map_resolution)]
                M = np.float32([[1, 0, self.full_w//2 - r], [0, 1, self.full_h//2 - c]])
                merge_map[e, i] = cv2.warpAffine(inputs[e, i], M, (self.full_w, self.full_h))
                #current_explored_map = cv2.warpAffine(current_explored_map, M, (self.size[0]*3, self.size[0]*3))
                # M = cv2.getRotationMatrix2D((self.full_w//2, self.full_h//2), self.full_pose[e, a, 2], 1) 
                # n_map[i] = cv2.warpAffine(n_move, M, (self.full_w, self.full_h)) 
                #current_explored_map = cv2.warpAffine(current_explored_map, M, (self.size[0]*3, self.size[0]*3)) 
        return merge_map

    def center_point_transform(self, goal_map, agent_id):
        point_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        merge_point_map = np.zeros((self.n_rollout_threads, 1, self.full_w, self.full_h), dtype=np.float32)
        trans_point = np.zeros((self.n_rollout_threads, 2))
        
        for e in range(self.n_rollout_threads):
            # for agent_id in range(self.num_agents):
            r, c = self.full_pose[e, agent_id,:2]
            r, c =[int(r * 100.0 / self.map_resolution), int(c * 100.0 / self.map_resolution)]
            M = np.float32([[1, 0, self.full_w//2 - r], [0, 1, self.full_h//2 - c]])
            point_map[e] = cv2.warpAffine(goal_map[e], M, (self.full_w, self.full_h))
            
            # map = torch.from_numpy(point_map[e])
            # n_rotated = F.grid_sample(map.unsqueeze(0).unsqueeze(0).float(), rotation[e][agent_id].float(), align_corners=True)
            # n_map = F.grid_sample(n_rotated.float(), trans[e][agent_id].float(), align_corners=True)
            # point_map[e] = n_map[0, 0].numpy()
            
            trans_point[e, 0], trans_point[e, 1] = np.unravel_index(np.argmax(point_map[e], axis=None), point_map[e].shape)
            
            merge_point_map[e, 0, int(trans_point[e, 0]-2): int(trans_point[e, 0]+3), int(trans_point[e, 1]-2): int(trans_point[e, 1]+3)] += 1
            
        #self.merge_goal_trace[:, agent_id, :, :] +=  merge_point_map[:, 0]
        #merge_point_map[:, 1] = self.merge_goal_trace[:, agent_id, :, :]
        
        return merge_point_map, trans_point
    
    # def point_transform(self, point, agent_id):
    #     merge_point_map = np.zeros((self.n_rollout_threads, 1, self.full_w, self.full_h), dtype=np.float32)
        
    #     for e in range(self.n_rollout_threads):
    #         merge_point_map[e, 0, int(point[e, agent_id, 0]*self.full_w-2): int(point[e, agent_id, 0]*self.full_w+3), \
    #             int(point[e, agent_id, 1]*self.full_h-2): int(point[e, agent_id, 1]*self.full_h+3)] += 1
            
    #     #self.merge_goal_trace[:, agent_id, :, :] +=  merge_point_map[:, 0]
    #     #merge_point_map[:, 1] = self.merge_goal_trace[:, agent_id, :, :]
        
    #     return merge_point_map
    
    def point_transform(self, agent_id):
        merge_point_map = np.zeros((self.n_rollout_threads, 2, self.full_w, self.full_h), dtype=np.float32)
        
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                goal_x = self.global_goal[e, a, 0]*(self.sim_map_size[e][a][0]+10)+self.center_w-self.sim_map_size[e][a][0]//2-5  
                goal_y = self.global_goal[e, a, 1]*(self.sim_map_size[e][a][1]+10)+self.center_h-self.sim_map_size[e][a][1]//2-5
                if self.direction_goal:
                    goal_x = self.global_goal[e, a, 0] * self.full_w
                    goal_y = self.global_goal[e, a, 1] * self.full_h
                if self.use_agent_id:    
                    merge_point_map[e, 0, int(goal_x-2): int(goal_x+3), int(goal_y-2): int(goal_y+3)] += (a + 1)/np.array([aa+1 for aa in range(self.num_agents)]).sum()
                else:
                    merge_point_map[e, 0, int(goal_x-2): int(goal_x+3), int(goal_y-2): int(goal_y+3)] += 1
            
        self.merge_goal_trace[:, agent_id] =  np.maximum(self.merge_goal_trace[:, agent_id], merge_point_map[:, 0])
        merge_point_map[:, 1] = self.merge_goal_trace[:, agent_id].copy()
        
        return merge_point_map

    def single_point_transform(self, a):
        merge_point_map = np.zeros((self.n_rollout_threads, 2, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            goal_x = self.global_goal[e, a, 0]*(self.sim_map_size[e][a][0]+10)+self.center_w-self.sim_map_size[e][a][0]//2-5  
            goal_y = self.global_goal[e, a, 1]*(self.sim_map_size[e][a][1]+10)+self.center_h-self.sim_map_size[e][a][1]//2-5
            if self.direction_goal:
                goal_x = self.global_goal[e, a, 0] * self.full_w
                goal_y = self.global_goal[e, a, 1] * self.full_h
            if self.use_agent_id:
                merge_point_map[e, 0, int(goal_x-2): int(goal_x+3), int(goal_y-2): int(goal_y+3)] = (a + 1)/np.array([aa+1 for aa in range(self.num_agents)]).sum()
            else:
                merge_point_map[e, 0, int(goal_x-2): int(goal_x+3), int(goal_y-2): int(goal_y+3)] = 1
            
            if self.grid_last_goal:
                self.grid_goal_pos[e, a, 0] = int((goal_x-(self.full_w/2-self.sim_map_size[e][a][0]//2-5))/((self.sim_map_size[e][a][0]+10)/self.grid_size))
                self.grid_goal_pos[e, a, 1] = int((goal_y-(self.full_h/2-self.sim_map_size[e][a][1]//2-5))/((self.sim_map_size[e][a][1]+10)/self.grid_size))
            
        self.merge_goal_trace[:, a] =  np.maximum(self.merge_goal_trace[:, a], merge_point_map[:, 0])
        merge_point_map[:, 1] = self.merge_goal_trace[:, a].copy()
        
        return merge_point_map

    '''def exp_transform(self, agent_id, inputs):
        explorable_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            r, c = self.full_pose[e, agent_id,:2]
            r, c =[int(r * 100.0 / self.map_resolution), int(c * 100.0 / self.map_resolution)]
            M = np.float32([[1, 0, self.full_w//2 - r], [0, 1, self.full_h//2 - c]])
            explorable_map[e] = cv2.warpAffine(inputs[e], M, (self.full_w, self.full_h))
        
        return explorable_map'''
    
    def center_gt_transform(self, inputs, a):
        merge_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            r, c = self.full_pose[e, a,:2]
            r, c =[int(r * 100.0 / self.map_resolution), int(c * 100.0 / self.map_resolution)]
            M = np.float32([[1, 0, self.full_w//2 - r], [0, 1, self.full_h//2 - c]])
            merge_map[e] = cv2.warpAffine(inputs[e], M, (self.full_w, self.full_h))
        return merge_map

    def compute_local_merge(self, inputs, single_map, a):
        local_merge_map = np.zeros((self.n_rollout_threads, 4, self.local_map_w, self.local_map_h))
        local_single_map = np.zeros((self.n_rollout_threads, 2, self.local_map_w, self.local_map_h))
        for e in range(self.n_rollout_threads):
            
            x = int(self.world_locs[e,a,0]-(self.center_w-self.sim_map_size[e][a][0]//2-5))
            y = int(self.world_locs[e,a,1]-(self.center_h-self.sim_map_size[e][a][1]//2-5))
            
            self.local_lmb[e, a] = self.get_local_map_boundaries((x, y), (self.local_map_w, self.local_map_h), ((self.sim_map_size[e][a][0]+10), (self.sim_map_size[e][a][1]+10)))
            sim_inputs = inputs[e, :, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: \
                                    self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5]
            local_merge_map[e] = sim_inputs[ :, self.local_lmb[e, a, 0] :self.local_lmb[e, a, 1], self.local_lmb[e, a, 2]:self.local_lmb[e, a, 3]]
            
            single_sim_inputs = single_map[e, :, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: \
                                    self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5]
            local_single_map[e] = single_sim_inputs[ :, self.local_lmb[e, a, 0] :self.local_lmb[e, a, 1], self.local_lmb[e, a, 2]:self.local_lmb[e, a, 3]]

            
        return local_merge_map, local_single_map
    
    def first_compute_global_input(self):
        locs = self.local_pose
        
        if self.use_single:
            self.single_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        else:
            self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        
        if self.use_merge_goal or self.use_goal:
            global_goal_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.full_w, self.full_h), dtype=np.float32)
        if self.use_local:
            self.local_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_map_w, self.local_map_h), dtype=np.float32)
            self.local_single_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_map_w, self.local_map_h), dtype=np.float32)
        if self.grid_pos:
            grid_pos = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        if self.grid_last_goal:
            grid_goal = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        for a in range(self.num_agents):
            for e in range(self.n_rollout_threads):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]
                                
                self.local_map[e, a, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1
                if self.use_orientation:
                    self.global_input['global_orientation'][e, a, 0] = int((locs[e, a, 2] + 180.0) / 5.)
                    self.other_agent_rotation[e, a, 0] = locs[e, a, 2]
                if self.use_vector_agent_id:
                    self.global_input['vector'][e, a] = np.eye(self.num_agents)[a]
            if self.use_single:
                self.single_merge_map[:, a] = self.single_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
            else:
                self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
            if self.use_local:
                self.local_merge_map[:, a], self.local_single_map[:, a] = self.compute_local_merge(self.merge_map[:, a], self.agent_merge_map[:,a], a)
            if self.use_merge_goal:
                global_goal_map[:, a] = self.point_transform(a)
            if self.use_goal:
                global_goal_map[:, a] = self.single_point_transform(a)
                
            #self.global_input['global_obs'][:, a, 0:4] = self.local_map[:, a].copy()
            #self.global_input['global_obs'][:, a, 0:4] = (nn.MaxPool2d(self.global_downscaling)(check(self.full_map[:, a]))).numpy()
            #_, self.trans_point[:,a] =self.point_transform(self.global_goal, np.array(self.agent_trans)[:, a], np.array(self.agent_rotation)[:,a], a)
            if self.use_single_agent_trace:
                if self.use_resnet:
                    for e in range(self.n_rollout_threads):
                        self.global_input['trace_image'][e, a, 0] = cv2.resize(self.agent_trace_map[e, a, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                        channel_i = 0
                        for agent_id in range(self.num_agents):
                            if agent_id != a:
                                channel_i += 1
                                self.global_input['trace_image'][e, a, channel_i] = cv2.resize(self.agent_trace_map[e, agent_id, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                else:
                    for e in range(self.n_rollout_threads):
                        if self.use_local:
                            self.global_input['trace_image'][e, a, 0] = self.local_single_map[e, a, 1]
                            if not self.use_own:
                                channel_i = 0
                                for agent_id in range(self.num_agents):
                                    if agent_id != a:
                                        channel_i += 1
                                        self.global_input['trace_image'][e, a, channel_i] = self.local_single_map[e, agent_id, 1]
                        else:
                            self.global_input['trace_image'][e, a, 0] = cv2.resize(self.agent_trace_map[e, a, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                            if not self.use_own:
                                channel_i = 0
                                for agent_id in range(self.num_agents):
                                    if agent_id != a:
                                        channel_i += 1
                                        self.global_input['trace_image'][e, a, channel_i] = cv2.resize(self.agent_trace_map[e, agent_id, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                            self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                            self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                
            if self.use_cnn_agent_id:
                if self.use_resnet:
                    self.global_input['vector_cnn'][:, a, 0] = np.ones((self.n_rollout_threads, 224, 224)) * ((a+1) /np.array([aa+1 for aa in range(self.num_agents)]).sum()) 
                    # for e in range(self.n_rollout_threads):
                    #     self.global_input['vector_cnn'][e, a, 1] = cv2.resize(self.agent_pos_map[e, a, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                    #     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                    #     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                    #     if not self.use_own:
                    #         channel_i = 0
                    #         for agent_id in range(self.num_agents):
                    #             if agent_id != a:
                    #                 channel_i += 1
                    #                 self.global_input['vector_cnn'][e, a, channel_i + 1] =  cv2.resize(self.agent_pos_map[e, agent_id, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                    #                     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                    #                     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                                            
                else:
                    if self.use_local:
                        self.global_input['vector_cnn'][:, a, 0] = np.ones((self.n_rollout_threads, self.local_map_w, self.local_map_h)) * ((a+1) /np.array([aa+1 for aa in range(self.num_agents)]).sum()) 
                        # for e in range(self.n_rollout_threads):
                        #     self.global_input['vector_cnn'][e, a, 1] = self.local_single_map[e, a, 0]
                        
                        #     if not self.use_own:
                        #         channel_i = 0
                        #         for agent_id in range(self.num_agents):
                        #             if agent_id != a:
                        #                 channel_i += 1
                        #                 self.global_input['vector_cnn'][e, a, channel_i + 1] = self.local_single_map[e, agent_id, 0]
                    else:    
                        self.global_input['vector_cnn'][:, a, 0] = np.ones((self.n_rollout_threads, self.local_w, self.local_h)) * ((a+1) /np.array([aa+1 for aa in range(self.num_agents)]).sum()) 
                        # for e in range(self.n_rollout_threads):
                        #     self.global_input['vector_cnn'][e, a, 1] = cv2.resize(self.agent_pos_map[e, a, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                        #     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                        #     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                        #     if not self.use_own:
                        #         channel_i = 0
                        #         for agent_id in range(self.num_agents):
                        #             if agent_id != a:
                        #                 channel_i += 1
                        #                 self.global_input['vector_cnn'][e, a, channel_i + 1] = cv2.resize(self.agent_pos_map[e, agent_id, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                        #                     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                        #                     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))

               
        if self.use_orientation:
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    i = 0
                    for l in range(self.num_agents):
                        if l!= a:   
                            if self.use_abs_orientation:
                                self.global_input['other_global_orientation'][e, a, i] = int((self.other_agent_rotation[e, l, 0] + 180.0) / 5.)
                            else:
                                self.global_input['other_global_orientation'][e, a, i] = int(((self.other_agent_rotation[e, l, 0] - self.other_agent_rotation[e, a, 0] + 360.0 + 180.0) % 360) / 5.)
                            i += 1
        if self.grid_pos: 
            for i in range(self.grid_size):
                grid_pos[:, 0:self.num_agents, i] = i - self.grid_locs[:, :, 0]
                grid_pos[:, self.num_agents:2*self.num_agents+1, i] = i - self.grid_locs[:, :,1]
            x_pos = np.expand_dims(grid_pos[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_pos'][ :, a, 0] = np.expand_dims(x_pos[ :, a, a], -1).repeat(self.grid_size, axis=-1)
            else:
                self.global_input['grid_pos'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_pos, -1).repeat(self.grid_size, axis=-1)
                
            y_pos = np.expand_dims(grid_pos[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_pos'][ :, a, 1] = np.expand_dims(y_pos[ :, a, a], -2).repeat(self.grid_size, axis=-2)
            else:    
                self.global_input['grid_pos'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_pos, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_pos'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_pos'][:, a] = np.roll(self.global_input['grid_pos'][:, a], -2*a, axis=1)
        
        if self.grid_last_goal: 
            for i in range(self.grid_size):
                grid_goal[:, 0:self.num_agents, i] = i - self.grid_goal_pos[:, :, 0]
                grid_goal[:, self.num_agents:2*self.num_agents+1, i] = i - self.grid_goal_pos[:, :,1]
            x_goal = np.expand_dims(grid_goal[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_goal'][ :, a, 0] = np.expand_dims(x_goal[ :, a, a], -1).repeat(self.grid_size, axis=-1)
            else:
                self.global_input['grid_goal'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_goal, -1).repeat(self.grid_size, axis=-1)
                
            y_goal = np.expand_dims(grid_goal[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_goal'][ :, a, 1] = np.expand_dims(y_goal[ :, a, a], -2).repeat(self.grid_size, axis=-2)
            else:    
                self.global_input['grid_goal'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_goal, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_goal'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_goal'][:, a] = np.roll(self.global_input['grid_goal'][:, a], -2*a, axis=1)
        
        for a in range(self.num_agents):# TODO @CHAO  
            if self.use_map_agent_id:
                map_agent_id = (a + 1)/np.array([aa + 1 for aa in range(self.num_agents)]).sum()
            else:
                map_agent_id = 1
            if self.use_centralized_V:
                world_gt = self.transform_gt(self.explorable_map[:, a], self.trans, self.rotation, a)
            for e in range(self.n_rollout_threads):
                if self.action_mask:
                    region_x_size = ((self.sim_map_size[e][a][0]+10)//self.grid_size)
                    region_y_size = ((self.sim_map_size[e][a][1]+10)//self.grid_size)
                    merge_map = self.merge_map[e,a,1,self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5].copy()       
                    merge_map += self.empty_map[e,a,self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5].copy()   
                    merge_map[merge_map>1]=1
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            if merge_map[i*region_x_size :(i+1)*region_x_size, j*region_y_size:(j+1)*region_y_size].sum() > region_x_size*region_y_size*0.98:
                                self.global_input['action_mask_obs'][e,a,0,i,j] = 0
                    
                if self.grid_agent_id:
                    self.global_input['grid_agent_id'][e, a, 0, self.grid_locs[e,a,0], self.grid_locs[e,a,1]] = 1
                
                if self.use_resnet:
                    for i in range(4):
                        if i > 1:
                            map_agent_id = 1
                        if self.use_single:
                            self.global_input['global_obs'][e, a, i] = cv2.resize(self.single_merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                            if self.use_single_merge and i<2:
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                        else:
                            if self.use_local:
                                if self.use_seperated_cnn_model:
                                    self.global_input['local_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.res_h, self.res_w))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w)) *map_agent_id
                                else:
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.res_h, self.res_w))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i+4] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                            else:
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                            
                        if self.use_goal and i<2:
                            self.global_input['global_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))      
                        
                        if self.use_merge_goal and i<2:
                            self.global_input['global_merge_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))                      
                    
                    if self.use_centralized_V:
                        self.share_global_input['gt_map'][e, a, 0] = cv2.resize(world_gt[e, 0, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))
                else:
                    for i in range(4):
                        if i > 1:
                            map_agent_id = 1
                        if self.use_single:
                            self.global_input['global_obs'][e, a, i] = cv2.resize(self.single_merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h)) * map_agent_id
                            if self.use_single_merge and i<2:
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h)) * map_agent_id
                            if self.use_goal and i<2:
                                self.global_input['global_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                        else:
                            if self.use_local:
                                if self.use_seperated_cnn_model:
                                    self.global_input['local_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.local_map_w, self.local_map_h))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))*map_agent_id
                                else:
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.local_map_w, self.local_map_h))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i+4] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))*map_agent_id
                                if self.use_merge_goal and i<2:
                                    self.global_input['global_merge_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))
                            else:   
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))*map_agent_id
                            
                                if self.use_merge_goal and i<2:
                                    self.global_input['global_merge_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                    
                    if self.use_centralized_V:
                        if self.use_local:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(world_gt[e, 0, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))
                        else:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(world_gt[e, 0, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h)) 
        
        if self.direction_goal:
            world_merge_map =  self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
            self.direction_map, self.direction_vec = self.goal_planner.prepare(world_merge_map, self.world_locs, self.n_rollout_threads, self.num_agents, self.env_step, envs = self.envs, parallel = True)
            if self.use_resnet:
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        for k in range(3 * self.all_args.direction_k + 1):
                            self.global_input['direction_map'][e, a, k] = cv2.resize(self.direction_map[e, a, k, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                            self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                            self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))
            else:
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        for k in range(3 * self.all_args.direction_k + 1):
                            self.global_input['direction_map'][e, a, k] = cv2.resize(self.direction_map[e, a, k, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                            self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                            self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
            self.global_input['direction_vec'] = self.direction_vec
        if self.action_mask:
            self.global_input['action_mask'] = self.global_input['action_mask_obs'].copy()
        
        all_global_cnn_input = [[] for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            for key in self.global_input.keys():
                if key not in ['stack_obs','global_orientation', 'vector','action_mask_obs','action_mask','grid_agent_id','grid_pos','grid_goal']:
                    all_global_cnn_input[agent_id].append(self.global_input[key][:, agent_id])
            all_global_cnn_input[agent_id] = np.concatenate(all_global_cnn_input[agent_id], axis=1) #[e,n,...]
        
        all_global_cnn_input = np.stack(all_global_cnn_input, axis=1)
        
        self.global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.local_w, self.local_h), dtype=np.float32)
        self.share_global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.local_w, self.local_h), dtype=np.float32)
        for agent_id in range(self.num_agents):
            self.global_input['stack_obs'][:, agent_id] = all_global_cnn_input.reshape(self.n_rollout_threads, -1, *all_global_cnn_input.shape[3:]).copy()
        
        for a in range(1, self.num_agents):
            self.global_input['stack_obs'][:, a] = np.roll(self.global_input['stack_obs'][:, a], -all_global_cnn_input.shape[2]*a, axis=1)

        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()
        
        self.first_compute = False
        
    def compute_local_action(self):
        local_action = torch.empty(self.n_rollout_threads, self.num_agents)
        for a in range(self.num_agents):
            local_goals = self.global_output[:, a, :-1]

            if self.train_local:
                torch.set_grad_enabled(True)
        
            action, action_prob, self.local_rnn_states[:, a] =\
                self.local_policy(self.obs[:, a],
                                    self.local_rnn_states[:, a],
                                    self.local_masks[:, a],
                                    extras=local_goals)

            if self.train_local:
                action_target = check(self.global_output[:, a, -1]).to(self.device)
                self.local_policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                torch.set_grad_enabled(False)
            
            local_action[:, a] = action.cpu()

        return local_action

    def rot_goals(self, e):
        trans_goals = np.zeros((self.num_agents, 2), dtype = np.int32)
        for agent_id in range(self.num_agents):
            goals = np.zeros((2), dtype = np.int32)
            if self.use_local:
                goals[0] = self.global_goal[e, agent_id, 0]*(self.local_map_w)+self.local_lmb[e, agent_id, 0]+self.full_w/2-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5  
                goals[1] = self.global_goal[e, agent_id, 1]*(self.local_map_h)+self.local_lmb[e, agent_id, 2]+self.full_h/2-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5
            else:  
                goals[0] = self.global_goal[e, agent_id, 0]*(self.sim_map_size[e][agent_id][0]+10)+self.full_w/2-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5  
                goals[1] = self.global_goal[e, agent_id, 1]*(self.sim_map_size[e][agent_id][1]+10)+self.full_h/2-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5
            if self.direction_goal:
                goals[0] = int(self.global_goal[e, agent_id, 0] * self.full_w)
                goals[1] = int(self.global_goal[e, agent_id, 1] * self.full_h)
            
            output = np.zeros((1, 1, self.full_w, self.full_h), dtype =np.float32)
            output[0, 0, goals[0]-1 : goals[0]+2, goals[1]-1:goals[1]+2] = 1
            agent_n_trans = F.grid_sample(torch.from_numpy(output).float(), self.agent_trans[e][agent_id].float(), align_corners = True)
            map = F.grid_sample(agent_n_trans.float(), self.agent_rotation[e][agent_id].float(), align_corners = True)[0, 0, :, :].numpy()
            (index_a, index_b) = np.unravel_index(np.argmax(map[ :, :], axis=None), map[ :, :].shape) # might be wrong!                
            trans_goals[agent_id] = np.array([index_a, index_b], dtype = np.float32)
        return trans_goals
    
    def rot_map(self, inputs, e, a, agent_trans, agent_rotation):
        agent_n_trans = F.grid_sample(torch.from_numpy(inputs).unsqueeze(0).float(), agent_trans[e][a].float(), align_corners=True)      
        trans_map = F.grid_sample(agent_n_trans.float(), agent_rotation[e][a].float(), align_corners=True)[0, :, :, :].numpy()
        return trans_map

    def compute_merge_map_boundary(self, e, a):
            return 0, self.full_w, 0, self.full_h

    def compute_local_input(self, map):
        self.global_insert = []
        for e in range(self.n_rollout_threads):
            p_input = defaultdict(list)
            self.revise_global_goal[e] = self.rot_goals(e)
            for a in range(self.num_agents):                
                lx, rx, ly, ry = self.compute_merge_map_boundary(e, a)
                p_input['goal'].append(self.revise_global_goal[e, a])
                trans_map = self.rot_map(map[e, a, 0:2], e, a, self.agent_trans, self.agent_rotation)
                p_input['map_pred'].append(trans_map[0, :, :].copy())
                p_input['exp_pred'].append(trans_map[1, :, :].copy())
                pose_pred = self.planner_pose_inputs[e, a].copy()
                pose_pred[3:] = np.array((lx, rx, ly, ry))
                p_input['pose_pred'].append(pose_pred) 
            self.global_insert.append(p_input)
    
    def compute_global_input(self):
        locs = self.local_pose
        if self.use_single:
            self.single_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        else:
            self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.trans_point = np.zeros((self.n_rollout_threads, self.num_agents, 2))
        if self.use_merge_goal or self.use_goal:
            global_goal_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.full_w, self.full_h), dtype=np.float32)
        if self.grid_pos:
            grid_pos = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        if self.grid_last_goal:
            grid_goal = np.zeros((self.n_rollout_threads, self.num_agents*2, self.grid_size), dtype=np.int32)
        if self.use_local:
            self.local_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_map_w, self.local_map_h), dtype=np.float32)
            self.local_single_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_map_w, self.local_map_h), dtype=np.float32)
        for a in range(self.num_agents):
            for e in range(self.n_rollout_threads):
                if self.use_vector_agent_id:
                    self.global_input['vector'][e, a] = np.eye(self.num_agents)[a]
                if self.use_orientation:
                    self.global_input['global_orientation'][e, a, 0] = int((locs[e, a, 2] + 180.0) / 5.)
                    self.other_agent_rotation[e, a, 0] = locs[e, a, 2]
            if self.use_single:
                self.single_merge_map[:, a] = self.single_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
            else:
                self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
            if self.use_local:
                self.local_merge_map[:, a], self.local_single_map[:, a] = self.compute_local_merge(self.merge_map[:, a], self.agent_merge_map[:,a], a)
            if self.use_merge_goal:
                global_goal_map[:, a] = self.point_transform(a)
            if self.use_goal:
                global_goal_map[:, a] = self.single_point_transform(a)
            
            #self.global_input['global_obs'][:, a, 0:4] = self.local_map[:, a]
            #self.global_input['global_obs'][:, a, 0:4] = (nn.MaxPool2d(self.global_downscaling)(check(self.full_map[:, a]))).numpy()
            #_, self.trans_point[:, a] =self.point_transform(self.global_goal, np.array(self.agent_trans)[:,a], np.array(self.agent_rotation)[:, a], a)
            if self.use_single_agent_trace:
                if self.use_resnet:
                    for e in range(self.n_rollout_threads):
                        self.global_input['trace_image'][e, a, 0] = cv2.resize(self.agent_trace_map[e, a, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                        channel_i = 0
                        for agent_id in range(self.num_agents):
                            if agent_id != a:
                                channel_i += 1
                                self.global_input['trace_image'][e, a, channel_i] = cv2.resize(self.agent_trace_map[e, agent_id, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                else:
                    for e in range(self.n_rollout_threads):
                        if self.use_local:
                                self.global_input['trace_image'][e, a, 0] = self.local_single_map[e, a, 1]
                                if not self.use_own:
                                    channel_i = 0
                                    for agent_id in range(self.num_agents):
                                        if agent_id != a:
                                            channel_i += 1
                                            self.global_input['trace_image'][e, a, channel_i] = self.local_single_map[e, agent_id, 1]
                        else:
                            self.global_input['trace_image'][e, a, 0] = cv2.resize(self.agent_trace_map[e, a, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                            if not self.use_own:
                                channel_i = 0
                                for agent_id in range(self.num_agents):
                                    if agent_id != a:
                                        channel_i += 1
                                        self.global_input['trace_image'][e, a, channel_i] = cv2.resize(self.agent_trace_map[e, agent_id, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                            self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                            self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
            if self.use_cnn_agent_id:
                if self.use_resnet:
                    self.global_input['vector_cnn'][:, a, 0] = np.ones((self.n_rollout_threads, 224, 224)) * ((a+1) /np.array([aa+1 for aa in range(self.num_agents)]).sum()) 
                    # for e in range(self.n_rollout_threads):
                    #     self.global_input['vector_cnn'][e, a, 1] = cv2.resize(self.agent_pos_map[e, a, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                    #     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                    #     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                    #     if not self.use_own:
                    #         channel_i = 0
                    #         for agent_id in range(self.num_agents):
                    #             if agent_id != a:
                    #                 channel_i += 1
                    #                 self.global_input['vector_cnn'][e, a, channel_i + 1] =  cv2.resize(self.agent_pos_map[e, agent_id, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                    #                     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                    #                     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_w, self.res_h))
                                            
                else:
                    
                    for e in range(self.n_rollout_threads):
                        if self.use_local:
                            self.global_input['vector_cnn'][:, a, 0] = np.ones((self.n_rollout_threads, self.local_map_w, self.local_map_h)) * ((a+1) /np.array([aa+1 for aa in range(self.num_agents)]).sum()) 
                            # for e in range(self.n_rollout_threads):
                            #     self.global_input['vector_cnn'][e, a, 1] = self.local_single_map[e, a, 0]
                            
                            #     if not self.use_own:
                            #         channel_i = 0
                            #         for agent_id in range(self.num_agents):
                            #             if agent_id != a:
                            #                 channel_i += 1
                            #                 self.global_input['vector_cnn'][e, a, channel_i + 1] = self.local_single_map[e, agent_id, 0]
                        else:
                            self.global_input['vector_cnn'][:, a, 0] = np.ones((self.n_rollout_threads, self.local_w, self.local_h)) * ((a+1) /np.array([aa+1 for aa in range(self.num_agents)]).sum()) 
                            # for e in range(self.n_rollout_threads):
                            #     self.global_input['vector_cnn'][e, a, 1] = cv2.resize(self.agent_pos_map[e, a, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                            #     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                            #     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                            #     if not self.use_own:
                            #         channel_i = 0
                            #         for agent_id in range(self.num_agents):
                            #             if agent_id != a:
                            #                 channel_i += 1
                            #                 self.global_input['vector_cnn'][e, a, channel_i + 1] = cv2.resize(self.agent_pos_map[e, agent_id, self.local_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                            #                     self.local_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                            #                     self.local_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.local_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))

        if self.use_orientation:
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    i = 0
                    for l in range(self.num_agents):
                        if l!= a:   
                            if self.use_abs_orientation:
                                self.global_input['other_global_orientation'][e, a, i] = int((self.other_agent_rotation[e, l, 0] + 180.0) / 5.)
                            else:
                                self.global_input['other_global_orientation'][e, a, i] = int(((self.other_agent_rotation[e, l, 0] - self.other_agent_rotation[e, a, 0] + 360.0 + 180.0) % 360) / 5.)
                            i += 1 
        if self.grid_pos: 
            for i in range(self.grid_size):
                grid_pos[:, 0:self.num_agents, i] = i - self.grid_locs[:, :, 0]
                grid_pos[:, self.num_agents:2*self.num_agents+1, i] = i - self.grid_locs[:, :,1]
            x_pos = np.expand_dims(grid_pos[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_pos'][ :, a, 0] = np.expand_dims(x_pos[ :, a, a], -1).repeat(self.grid_size, axis=-1)
            else:
                self.global_input['grid_pos'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_pos, -1).repeat(self.grid_size, axis=-1)
                
            y_pos = np.expand_dims(grid_pos[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_pos'][ :, a, 1] = np.expand_dims(y_pos[ :, a, a], -2).repeat(self.grid_size, axis=-2)
            else:    
                self.global_input['grid_pos'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_pos, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_pos'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_pos'][:, a] = np.roll(self.global_input['grid_pos'][:, a], -2*a, axis=1)
        
        if self.grid_last_goal: 
            for i in range(self.grid_size):
                grid_goal[:, 0:self.num_agents, i] = i - self.grid_goal_pos[:, :, 0]
                grid_goal[:, self.num_agents:2*self.num_agents+1, i] = i - self.grid_goal_pos[:, :,1]
            x_goal = np.expand_dims(grid_goal[:, 0:self.num_agents], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_goal'][ :, a, 0] = np.expand_dims(x_goal[ :, a, a], -1).repeat(self.grid_size, axis=-1)
            else:
                self.global_input['grid_goal'][ :, :, 0:2*self.num_agents:2] = np.expand_dims(x_goal, -1).repeat(self.grid_size, axis=-1)
                
            y_goal = np.expand_dims(grid_goal[:, self.num_agents:2*self.num_agents+1], 1).repeat(self.num_agents, axis=1)
            if self.use_self_grid_pos:
                for a in range(self.num_agents):
                    self.global_input['grid_goal'][ :, a, 1] = np.expand_dims(y_goal[ :, a, a], -2).repeat(self.grid_size, axis=-2)
            else:    
                self.global_input['grid_goal'][:, :, 1:2*self.num_agents:2] = np.expand_dims(y_goal, -2).repeat(self.grid_size, axis=-2)
            self.global_input['grid_goal'] += self.grid_size - 1 #(0-14)
            if self.agent_invariant:
                for a in range(1, self.num_agents):
                    self.global_input['grid_goal'][:, a] = np.roll(self.global_input['grid_goal'][:, a], -2*a, axis=1)
           
        for a in range(self.num_agents):
            if self.use_map_agent_id:
                map_agent_id = (a + 1)/np.array([aa + 1 for aa in range(self.num_agents)]).sum()
            else:
                map_agent_id = 1
            if self.use_centralized_V:
                world_gt = self.transform_gt(self.explorable_map[:, a], self.trans, self.rotation, a)
            for e in range(self.n_rollout_threads):
                if self.action_mask:
                    region_x_size = ((self.sim_map_size[e][a][0]+10)//self.grid_size)
                    region_y_size = ((self.sim_map_size[e][a][1]+10)//self.grid_size)
                    merge_map = self.merge_map[e,a,1,self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5].copy()          
                    merge_map += self.empty_map[e,a,self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5].copy()   
                    merge_map[merge_map>1]=1
                    
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            if merge_map[i*region_x_size :(i+1)*region_x_size, j*region_y_size:(j+1)*region_y_size].sum() > region_x_size*region_y_size*0.98:
                                self.global_input['action_mask_obs'][e,a,0,i,j] = 0
                    
                    if self.env_info['sum_merge_explored_ratio'][e]> self.all_args.explored_ratio_down_threshold:
                        self.global_input['action_mask'][e,a] = 1
                    else:
                        self.global_input['action_mask'][e,a] = self.global_input['action_mask_obs'][e,a].copy()
                               
                if self.use_resnet:
                    for i in range(4):
                        if i > 1:
                            map_agent_id = 1
                        if self.use_single:
                            self.global_input['global_obs'][e, a, i] = cv2.resize(self.single_merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                            if self.use_single_merge and i<2:
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                        else:
                            if self.use_local:
                                if self.use_seperated_cnn_model:
                                    self.global_input['local_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.res_h, self.res_w))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w)) *map_agent_id
                                else:
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.res_h, self.res_w))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i+4] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                            else:
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))*map_agent_id
                            
                        if self.use_goal and i<2:
                            self.global_input['global_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))          
                        if self.use_merge_goal and i<2:
                            self.global_input['global_merge_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))                  
                    if self.use_centralized_V:
                        self.share_global_input['gt_map'][e, a, 0] = cv2.resize(world_gt[e, 0, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.res_h, self.res_w))
                else:
                    for i in range(4):
                        if i > 1:
                            map_agent_id = 1
                        if self.use_single:
                            self.global_input['global_obs'][e, a, i] = cv2.resize(self.single_merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))*map_agent_id
                            if self.use_single_merge and i<2:
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))*map_agent_id
                            if self.use_goal and i<2:
                                    self.global_input['global_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                        else:
                            if self.use_local:
                                if self.use_seperated_cnn_model:
                                    self.global_input['local_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.local_map_w, self.local_map_h))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))*map_agent_id
                                else:
                                    self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.local_map_w, self.local_map_h))*map_agent_id
                                    self.global_input['global_merge_obs'][e, a, i+4] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                        self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                        self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))*map_agent_id
                                if self.use_merge_goal and i<2:
                                    self.global_input['global_merge_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))
                            else:   
                                self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.merge_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))*map_agent_id
                            
                                if self.use_merge_goal and i<2:
                                    self.global_input['global_merge_goal'][e, a, i] = cv2.resize(global_goal_map[e, a, i, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h))
                    if self.use_centralized_V:
                        if self.use_local:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(world_gt[e, 0, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_map_w, self.local_map_h))
                        else:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(world_gt[e, 0, self.center_w-math.ceil(self.sim_map_size[e][a][0]/2)-5: \
                                    self.center_w+math.ceil(self.sim_map_size[e][a][0]/2)+5, \
                                    self.center_h-math.ceil(self.sim_map_size[e][a][1]/2)-5: self.center_h+math.ceil(self.sim_map_size[e][a][1]/2)+5], (self.local_w, self.local_h)) 

        all_global_cnn_input = [[] for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            for key in self.global_input.keys():
                if key not in ['stack_obs','global_orientation', 'vector','action_mask_obs','action_mask','grid_agent_id','grid_pos','grid_goal']:

                    all_global_cnn_input[agent_id].append(self.global_input[key][:, agent_id])
            all_global_cnn_input[agent_id] = np.concatenate(all_global_cnn_input[agent_id], axis=1) #[e,n,...]
        
        all_global_cnn_input = np.stack(all_global_cnn_input, axis=1)

        self.global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.local_w, self.local_h), dtype=np.float32)
        self.share_global_input['stack_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.local_w, self.local_h), dtype=np.float32)
        
        for agent_id in range(self.num_agents):
            self.global_input['stack_obs'][:, agent_id] = all_global_cnn_input.reshape(self.n_rollout_threads, -1, *all_global_cnn_input.shape[3:]).copy()
        
        for a in range(1, self.num_agents):
            self.global_input['stack_obs'][:, a] = np.roll(self.global_input['stack_obs'][:,a], -all_global_cnn_input.shape[2]*a, axis=1)

        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()
        
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, self.share_global_input)
    
    def goal_to_frontier(self):
        merge_map =  self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                goals = np.zeros((2), dtype = np.int32)
                if self.use_local:
                    goals[0] = self.global_goal[e, agent_id, 0]*(self.local_map_w)+self.local_lmb[e, agent_id, 0]+self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5  
                    goals[1] = self.global_goal[e, agent_id, 1]*(self.local_map_h)+self.local_lmb[e, agent_id, 2]+self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5
                else:  
                    goals[0] = self.global_goal[e, agent_id, 0]*(self.sim_map_size[e][agent_id][0]+10)+self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5  
                    goals[1] = self.global_goal[e, agent_id, 1]*(self.sim_map_size[e][agent_id][1]+10)+self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5
                if self.direction_goal:
                    goals[0] = int(self.global_goal[e, agent_id, 0] * self.full_w)
                    goals[1] = int(self.global_goal[e, agent_id, 1] * self.full_h)
                
                goals = get_closest_frontier(merge_map[e], self.world_locs[e, agent_id], goals)

                if self.use_local:
                    self.global_goal[e, agent_id, 0] = (goals[0] - (self.local_lmb[e, agent_id, 0]+self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5)) / self.local_map_w
                    self.global_goal[e, agent_id, 1] = (goals[1] - (self.local_lmb[e, agent_id, 2]+self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5)) / self.local_map_h
                else:
                    self.global_goal[e, agent_id, 0] = (goals[0] - (self.center_w-math.ceil(self.sim_map_size[e][agent_id][0]/2)-5)) / (self.sim_map_size[e][agent_id][0]+10)
                    self.global_goal[e, agent_id, 1] = (goals[1] - (self.center_h-math.ceil(self.sim_map_size[e][agent_id][1]/2)-5)) / (self.sim_map_size[e][agent_id][1]+10)
                if self.direction_goal:
                    self.global_goal[e, agent_id, 0] = goals[0] / self.full_w
                    self.global_goal[e, agent_id, 1] = goals[1] / self.full_h
                
                self.global_goal[e, agent_id, 0] = max(0, min(1, self.global_goal[e, agent_id, 0]))
                self.global_goal[e, agent_id, 1] = max(0, min(1, self.global_goal[e, agent_id, 1]))

    def compute_global_goal(self, step):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][step])
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.concatenate(self.buffer.obs[key][step])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                            concat_obs,
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            rank = np.concatenate(self.buffer.rank[step]))
        
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # Compute planner inputs
        if self.direction_goal:
            world_merge_map =  self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
            self.global_goal = self.goal_planner.get_goal(world_merge_map, self.world_locs, actions, self.n_rollout_threads, self.num_agents, self.env_step, envs = self.envs, parallel = True)
        elif self.grid_goal:
            action = action.detach().clone()
            action[:, 1:] = nn.Sigmoid()(action[:, 1:])
            action = np.array(np.split(_t2n(action), self.n_rollout_threads))
            r, c = action[:, :, 0].astype(np.int32) // self.grid_size, action[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = (action[:, :, 1] + r) / self.grid_size
            self.global_goal[:, :, 1] = (action[:, :, 2] + c) / self.grid_size
        elif self.grid_goal_simpler:
            action = action.detach().clone()
            action = np.array(np.split(_t2n(action), self.n_rollout_threads))
            r, c = action[:, :, 0].astype(np.int32) // self.grid_size, action[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = (0.5 + r) / self.grid_size
            self.global_goal[:, :, 1] = (0.5 + c) / self.grid_size
        elif self.discrete_goal:
            r, c = actions[:, :, 0].astype(np.int32) // self.grid_size, actions[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = (0.5 + r) / self.grid_size
            self.global_goal[:, :, 1] = (0.5 + c) / self.grid_size
        elif self.multidiscrete_goal:
            self.global_goal = (actions + 0.5) / self.grid_size
        else:
            self.global_goal = np.array(np.split(_t2n(nn.Sigmoid()(action)), self.n_rollout_threads))
 
        if self.proj_frontier:
            self.goal_to_frontier()
            
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def eval_compute_global_goal(self, rnn_states):      
        self.trainer.prep_rollout()

        concat_obs = {}
        for key in self.global_input.keys():
            concat_obs[key] = np.concatenate(self.global_input[key])
        
        # print(self.all_args.direction_greedy)
        if self.direction_goal and self.all_args.direction_greedy:
            actions = self.goal_planner.greedy(self.global_input, self.n_rollout_threads, self.num_agents)
            actions = torch.from_numpy(np.concatenate(actions))
        else:
            actions, rnn_states = self.trainer.policy.act(concat_obs,
                                        np.concatenate(rnn_states),
                                        np.concatenate(self.global_masks),
                                        deterministic=True)
            
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        # Compute planner inputs
        if self.direction_goal:
            world_merge_map =  self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
            actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            self.global_goal = self.goal_planner.get_goal(world_merge_map, self.world_locs, actions, self.n_rollout_threads, self.num_agents, self.env_step, envs = self.envs, parallel = True)
        elif self.grid_goal:
            actions = actions.detach().clone()
            actions[:, 1:] = nn.Sigmoid()(actions[:, 1:])
            action = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            r, c = action[:, :, 0].astype(np.int32) // self.grid_size, action[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = (action[:, :, 1] + r) / self.grid_size
            self.global_goal[:, :, 1] = (action[:, :, 2] + c) / self.grid_size
        elif self.grid_goal_simpler:
            actions = actions.detach().clone()
            action = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            r, c = action[:, :, 0].astype(np.int32) // self.grid_size, action[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = (0.5 + r) / self.grid_size
            self.global_goal[:, :, 1] = (0.5 + c) / self.grid_size
        elif self.discrete_goal:
            actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            r, c = actions[:, :, 0].astype(np.int32) // self.grid_size, actions[:, :, 0].astype(np.int32) % self.grid_size
            self.global_goal[:, :, 0] = (0.5 + r) / self.grid_size
            self.global_goal[:, :, 1] = (0.5 + c) / self.grid_size
        elif self.multidiscrete_goal:
            actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            self.global_goal = (actions + 0.5) / self.grid_size
        else:
            self.global_goal = np.array(np.split(_t2n(nn.Sigmoid()(actions)), self.n_rollout_threads))

        if self.proj_frontier:
            self.goal_to_frontier() 

        return rnn_states
    
    def eval_compute_single_global_goal(self, e, a, rnn_states):      
        
        self.trainer.prep_rollout()

        concat_obs = {}
        for key in self.global_input.keys():
            concat_obs[key] = self.global_input[key][e, a][np.newaxis]
        
        # print(self.all_args.direction_greedy)
        if self.direction_goal and self.all_args.direction_greedy:
            actions = self.goal_planner.greedy(self.global_input, self.n_rollout_threads, self.num_agents)
            actions = torch.from_numpy(np.concatenate(actions))
        else:
            actions, agent_rnn_states = self.trainer.policy.act(concat_obs,
                                        rnn_states[e, a][np.newaxis],
                                        self.global_masks[e, a][np.newaxis],
                                        deterministic=True)
            
            rnn_states[e, a] = np.array(_t2n(agent_rnn_states.squeeze(0)))
            
        # Compute planner inputs
        if self.direction_goal:
            world_merge_map =  self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, -1)
            actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            self.global_goal = self.goal_planner.get_goal(world_merge_map, self.world_locs, actions, self.n_rollout_threads, self.num_agents, self.env_step, envs = self.envs, parallel = True)
        elif self.grid_goal:
            # Compute planner inputs
            actions = actions.detach().clone()
            action = actions.squeeze(0)
            action[1:] = nn.Sigmoid()(action[1:])
            action = np.array(action.cpu())
            r, c = action[0].astype(np.int32) // self.grid_size, action[0].astype(np.int32) % self.grid_size
            self.global_goal[e, a, 0] = (action[1] + r) / self.grid_size
            self.global_goal[e, a, 1] = (action[2] + c) / self.grid_size
        elif self.grid_goal_simpler:
            actions = actions.detach().clone()
            action = actions.squeeze(0)
            action = np.array(action.cpu())
            r, c = action[0].astype(np.int32) // self.grid_size, action[0].astype(np.int32) % self.grid_size
            self.global_goal[e, a, 0] = (0.5 + r) / self.grid_size
            self.global_goal[e, a, 1] = (0.5 + c) / self.grid_size
        elif self.discrete_goal:
            actions = np.array(actions.squeeze(0).cpu())
            r, c = actions[0].astype(np.int32) // self.grid_size, actions[0].astype(np.int32) % self.grid_size
            self.global_goal[e, a, 0] = (0.5 + r) / self.grid_size
            self.global_goal[e, a, 1] = (0.5 + c) / self.grid_size
        elif self.multidiscrete_goal:
            actions = np.array(actions.squeeze(0).cpu())
            self.global_goal[e, a] = (actions + 0.5) / self.grid_size
        else:
            self.global_goal[e, a] = np.array(_t2n(nn.Sigmoid()(actions).squeeze(0)))
        
        return rnn_states

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        concat_share_obs = {}
        concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.concatenate(self.buffer.obs[key][-1])

        next_values = self.trainer.policy.get_values(concat_share_obs,
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]),
                                                rank=np.concatenate(self.buffer.rank[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def update_local_map(self):        
        locs = self.local_pose
        self.planner_pose_inputs[:, :, :3] = locs + self.origins
        self.local_map[:, :, 2, :, :].fill(0.)  # Resetting current location channel
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.local_map[e, a, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1
                if self.use_new_trace and self.use_weight_trace:
                    self.local_map[e, a, 3] *= self.decay_weight

    def update_map_and_pose(self, update = True):
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]] = self.local_map[e, a]
                if update:
                    if self.use_new_trace:
                        self.full_map[e, a, 3] = np.zeros((self.full_w, self.full_h), dtype=np.float32)
                        self.full_map[e, a, 3, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]] = self.local_map[e, a, 3]
                    self.full_pose[e, a] = self.local_pose[e, a] + self.origins[e, a]

                    locs = self.full_pose[e, a]
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                    int(c * 100.0 / self.map_resolution)]

                    self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                        (self.local_w, self.local_h),
                                                        (self.full_w, self.full_h))

                    self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a].copy()
                    self.origins[e, a] = [self.lmb[e, a][2] * self.map_resolution / 100.0,
                                    self.lmb[e, a][0] * self.map_resolution / 100.0, 0.]

                    self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
                    self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]
                    if self.use_new_trace:
                        self.local_map[e, a, 3] = np.zeros((self.local_w, self.local_h), dtype=np.float32)
    
    def update_agent_map_and_pose(self, e, a):
        if self.use_new_trace:
            self.full_map[e, a, 3] = np.zeros((self.full_w, self.full_h), dtype=np.float32)
        self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]] = self.local_map[e, a]
        self.full_pose[e, a] = self.local_pose[e, a] + self.origins[e, a]

        locs = self.full_pose[e, a]
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                        int(c * 100.0 / self.map_resolution)]

        self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                            (self.local_w, self.local_h),
                                            (self.full_w, self.full_h))

        self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a].copy()
        self.origins[e, a] = [self.lmb[e, a][2] * self.map_resolution / 100.0,
                        self.lmb[e, a][0] * self.map_resolution / 100.0, 0.]

        self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
        self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]
        if self.use_new_trace:
            self.local_map[e, a, 3] = np.zeros((self.local_w, self.local_h), dtype=np.float32)

    def insert_global_policy(self, data):
        dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
    
        for e in range(self.n_rollout_threads):
            if self.use_intrinsic_reward and self.env_info['sum_merge_repeat_ratio'][e] >= 0.9:
                for agent_id in range(self.num_agents):
                    intrinsic_gt = self.intrinsic_gt[e , agent_id].copy()
                    intrinsic_gt[intrinsic_gt<self.map_threshold] = -1
                    intrinsic_gt[intrinsic_gt>=self.map_threshold] = 1
                    
                    reward_map = intrinsic_gt - self.merge_map[e, agent_id, 1]
                    if reward_map[self.revise_global_goal[e, agent_id, 0], self.revise_global_goal[e, agent_id, 1]] > 0.5:
                        self.rewards[e, agent_id] += 0.02
            if self.use_goal_penalty:
                for p_a in range(self.num_agents):
                    p_a_x = self.global_goal[e, p_a, 0]*self.grid_size
                    p_a_y = self.global_goal[e, p_a, 1]*self.grid_size
                    for p_b in range(self.num_agents):
                        if p_b != p_a:
                            p_b_x = self.global_goal[e, p_b, 0]*self.grid_size
                            p_b_y = self.global_goal[e, p_b, 1]*self.grid_size
                            if abs(p_a_x - p_b_x) + abs(p_a_y - p_b_y) < 2:
                                self.rewards[e, p_a] -= 0.05
            
            if self.use_pos_penalty:
                for p_a in range(self.num_agents):
                    p_a_x = self.world_locs[e, p_a, 0]
                    p_a_y = self.world_locs[e, p_a, 1]
                    for p_b in range(self.num_agents):
                        if p_b != p_a:
                            p_b_x = self.world_locs[e, p_b, 0]
                            p_b_y = self.world_locs[e, p_b, 1]
                            if abs(p_a_x - p_b_x) + abs(p_a_y - p_b_y) < 20:
                                self.rewards[e, p_a] -= 0.01
                
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        if not self.use_centralized_V:
            self.share_global_input = self.global_input.copy()

        self.buffer.insert(self.share_global_input, self.global_input, rnn_states, rnn_states_critic, actions, action_log_probs, values, self.rewards, self.global_masks)
        
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        if self.use_delta_reward:
            self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
    
    def ft_compute_global_goal(self, e):
        #print("    apf {}".format(e))
        # ft_merge_map transform
        locations = self.update_ft_merge_map(e)
        
        inputs = {
            'map_pred' : self.ft_merge_map[e,0],
            'exp_pred' : self.ft_merge_map[e,1],
            'locations' : locations
        }
        if self.all_args.algorithm_name == "ft_wma_rrt":
            goal_mask = [False for agent_id in range(self.num_agents)]
        else:
            goal_mask = [self.ft_go_steps[e][agent_id]<15 for agent_id in range(self.num_agents)]

        num_choose = self.num_agents - sum(goal_mask)
        self.env_info['merge_global_goal_num'] += num_choose

        goals = self.ft_get_goal(inputs, goal_mask, pre_goals = self.ft_pre_goals[e], e=e)

        for agent_id in range(self.num_agents):
            if not goal_mask[agent_id] or 'utility' in self.all_args.algorithm_name:
                self.ft_pre_goals[e][agent_id] = np.array(goals[agent_id], dtype=np.int32) # goals before rotation

        self.ft_goals[e]=self.rot_ft_goals(e, goals, goal_mask)
    
    def rot_ft_goals(self, e, goals, goal_mask = None):
        if goal_mask == None:
            goal_mask = [True for _ in range(self.num_agents)]
        ft_goals = np.zeros((self.num_agents, 2), dtype = np.int32)
        for agent_id in range(self.num_agents):
            if goal_mask[agent_id]:
                ft_goals[agent_id] = self.ft_goals[e][agent_id]
                continue
            self.ft_go_steps[e][agent_id] = 0
            output = np.zeros((1, 1, self.full_w, self.full_h), dtype =np.float32)
            output[0, 0, goals[agent_id][0]-1 : goals[agent_id][0]+2, goals[agent_id][1]-1:goals[agent_id][1]+2] = 1
            agent_n_trans = F.grid_sample(torch.from_numpy(output).float(), self.agent_trans[e][agent_id].float(), align_corners = True)
            map = F.grid_sample(agent_n_trans.float(), self.agent_rotation[e][agent_id].float(), align_corners = True)[0, 0, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(map[ :, :], axis=None), map[ :, :].shape) # might be wrong!!
            ft_goals[agent_id] = np.array([index_a, index_b], dtype = np.float32)
        return ft_goals
    
    def compute_merge_map_boundary(self, e, a, ft = True):
        return 0, self.full_w, 0, self.full_h

    def ft_compute_local_input(self):
        self.local_input = []

        for e in range(self.n_rollout_threads):
            p_input = defaultdict(list)
            for a in range(self.num_agents):
                lx, rx, ly, ry = self.compute_merge_map_boundary(e, a)
                p_input['goal'].append([int(self.ft_goals[e, a][0])-lx, int(self.ft_goals[e,a][1])-ly])
                if self.use_local_single_map:
                    p_input['map_pred'].append(self.full_map[e, a, 0, :, :].copy())
                    p_input['exp_pred'].append(self.full_map[e, a, 1, :, :].copy())
                else:
                    trans_map = self.rot_map(self.merge_map[e, a, 0:2], e, a, self.agent_trans, self.agent_rotation)
                    p_input['map_pred'].append(trans_map[0, :, :].copy())
                    p_input['exp_pred'].append(trans_map[1, :, :].copy())
                pose_pred = self.planner_pose_inputs[e, a].copy()
                pose_pred[3:] = np.array((lx, rx, ly, ry))
                p_input['pose_pred'].append(pose_pred)
            self.local_input.append(p_input)
            
    def update_ft_merge_map(self, e):
        full_map = self.full_map[e]
        self.ft_merge_map[e] = np.zeros((2, self.full_w, self.full_h), dtype = np.float32) # only explored and obstacle
        locations = []
        for agent_id in range(self.num_agents):
            output = torch.from_numpy(full_map[agent_id].copy())
            n_rotated = F.grid_sample(output.unsqueeze(0).float(), self.rotation[e][agent_id].float(), align_corners=True)
            n_map = F.grid_sample(n_rotated.float(), self.trans[e][agent_id].float(), align_corners = True)
            agent_merge_map = n_map[0, :, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[2, :, :], axis=None), agent_merge_map[2, :, :].shape) # might be inaccurate !!        
            self.ft_merge_map[e] += agent_merge_map[:2]
            locations.append((index_a, index_b))

        for i in range(2):
            self.ft_merge_map[e,i][self.ft_merge_map[e,i] > 1] = 1
            self.ft_merge_map[e,i][self.ft_merge_map[e,i] < self.map_threshold] = 0
        
        return locations
    
    def ft_get_goal(self, inputs, goal_mask, pre_goals = None, e=None):
        obstacle = inputs['map_pred']
        explored = inputs['exp_pred']
        locations = inputs['locations']

        if all(goal_mask):
            goals = []
            for agent_id in range(self.num_agents):
                goals.append((self.ft_pre_goals[e,agent_id][0], self.ft_pre_goals[e, agent_id][1]))
            return goals

        obstacle = np.rint(obstacle).astype(np.int32)
        explored = np.rint(explored).astype(np.int32)
        explored[obstacle == 1] = 1

        H, W = explored.shape
        steps = [(-1,0),(1,0),(0,-1),(0,1)]
        map, (lx, ly), unexplored = get_frontier(obstacle, explored, locations, cut_boundary=(self.all_args.algorithm_name != "ft_wma_rrt"))
        '''
        map: H x W
            - 0 for explored & available cell
            - 1 for obstacle
            - 2 for target (frontier)
        '''
        self.ft_map[e] = map.copy()
        self.ft_lx[e] = lx
        self.ft_ly[e] = ly
        
        goals = []
        locations = [(x-lx, y-ly) for x, y in locations]
        if self.all_args.algorithm_name in ['ft_wma_rrt']:
            if not hasattr(self, "wma_rrt") or self.env_step == 0:
                self.wma_rrt = WMA_RRT(
                    start=locations,
                    rand_area=((0, H), (0, W)),
                    expand_dis=20,
                    node_radius=15,
                    max_iter=2000,
                    map=map, 
                    unexplored=unexplored,
                    strict=self.all_args.wma_rrt_strict)
            self.wma_rrt.update_map(map, unexplored, locations)
            goals = self.wma_rrt.move(rrt_expand=(self.env_step % 5 == 0))
            goals = [(int(x), int(y)) for x, y in goals]
            print(goals)
        elif self.all_args.algorithm_name in ['ft_utility', 'ft_voronoi']:
            pre_goals = pre_goals.copy()
            pre_goals[:, 0] -= lx
            pre_goals[:, 1] -= ly
            if self.all_args.algorithm_name == 'ft_utility':
                goals = max_utility_frontier(map, unexplored, locations, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, utility_radius = self.all_args.utility_radius, pre_goals = pre_goals, goal_mask = goal_mask, random_goal=self.all_args.ft_use_random)
            elif self.all_args.algorithm_name == 'ft_voronoi':
                goals = voronoi_based_planning(map, unexplored, locations, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, utility_radius = self.all_args.utility_radius, pre_goals = pre_goals, goal_mask = goal_mask, random_goal=self.all_args.ft_use_random)
            goals[:, 0] += lx
            goals[:, 1] += ly
        else:
            for agent_id in range(self.num_agents): # replan when goal is not target
                if goal_mask[agent_id]:
                    goals.append((-1,-1))
                    continue
                if self.all_args.algorithm_name == 'ft_apf':
                    apf = APF(self.all_args)
                    path = apf.schedule(map, unexplored, locations, steps, agent_id, clear_disk = True, random_goal=self.all_args.ft_use_random)
                    goal = path[-1]
                elif self.all_args.algorithm_name == 'ft_nearest':
                    goal = nearest_frontier(map, unexplored, locations, steps, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, random_goal=self.all_args.ft_use_random)
                elif self.all_args.algorithm_name == 'ft_rrt':
                    goal = rrt_global_plan(map, unexplored, locations, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, step = self.env_step, utility_radius = self.all_args.utility_radius, random_goal=self.all_args.ft_use_random)
                else:
                    raise NotImplementedError
                goals.append((goal[0] + lx, goal[1] + ly))

        return goals

    def train_slam_module(self):
        for _ in range(self.slam_iterations):
            inputs, outputs = self.slam_memory.sample(self.slam_batch_size)
            b_obs_last, b_obs, b_poses = inputs
            gt_fp_projs, gt_fp_explored, gt_pose_err = outputs
            
            b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                self.nslam_module(b_obs_last, b_obs, b_poses,
                            None, None, None,
                            build_maps=False)
            
            gt_fp_projs = check(gt_fp_projs).to(self.device)
            gt_fp_explored = check(gt_fp_explored).to(self.device)
            gt_pose_err = check(gt_pose_err).to(self.device)
            
            loss = 0
            if self.proj_loss_coeff > 0:
                proj_loss = F.binary_cross_entropy(b_proj_pred.double(), gt_fp_projs.double())
                self.train_slam_infos['costs'].append(proj_loss.item())
                loss += self.proj_loss_coeff * proj_loss

            if self.exp_loss_coeff > 0:
                exp_loss = F.binary_cross_entropy(b_fp_exp_pred.double(), gt_fp_explored.double())
                self.train_slam_infos['exp_costs'].append(exp_loss.item())
                loss += self.exp_loss_coeff * exp_loss

            if self.pose_loss_coeff > 0:
                pose_loss = torch.nn.MSELoss()(b_pose_err_pred.double(), gt_pose_err.double())
                self.train_slam_infos['pose_costs'].append(self.pose_loss_coeff * pose_loss.item())
                loss += self.pose_loss_coeff * pose_loss
            
            self.slam_optimizer.zero_grad()
            loss.backward()
            self.slam_optimizer.step()

            del b_obs_last, b_obs, b_poses
            del gt_fp_projs, gt_fp_explored, gt_pose_err
            del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

    def train_local_policy(self):
        self.local_optimizer.zero_grad()
        self.local_policy_loss.backward()
        self.train_local_infos['local_policy_loss'].append(self.local_policy_loss.item())
        self.local_optimizer.step()
        self.local_policy_loss = 0
        self.local_rnn_states = self.local_rnn_states.detach_()

    def train_global_policy(self):
        self.compute()
        train_global_infos = self.train()
        
        for k, v in train_global_infos.items():
            self.train_global_infos[k].append(v)

    def save_slam_model(self, step):
        if self.train_slam:
            if len(self.train_slam_infos['costs']) >= 1000 and np.mean(self.train_slam_infos['costs']) < self.best_slam_cost:
                self.best_slam_cost = np.mean(self.train_slam_infos['costs'])
                torch.save(self.nslam_module.state_dict(), str(self.save_dir) + "/slam_best.pt")
            torch.save(self.nslam_module.state_dict(), str(self.save_dir) + "slam_periodic_{}.pt".format(step))

    def save_local_model(self, step):
        if self.train_local:
            if len(self.train_local_infos['local_policy_loss']) >= 100 and \
                (np.mean(self.train_local_infos['local_policy_loss']) <= self.best_local_loss):
                self.best_local_loss = np.mean(self.train_local_infos['local_policy_loss'])
                torch.save(self.local_policy.state_dict(), str(self.save_dir) + "/local_best.pt")
            torch.save(self.local_policy.state_dict(), str(self.save_dir) + "local_periodic_{}.pt".format(step))
    
    def save_global_model(self, step):
        if len(self.env_infos["sum_merge_explored_reward"]) >= self.all_args.eval_episodes and \
            (np.mean(self.env_infos["sum_merge_explored_reward"]) >= self.best_gobal_reward):
            self.best_gobal_reward = np.mean(self.env_infos["sum_merge_explored_reward"])
            torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_best.pt")
            torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_best.pt")
            torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/global_actor_optimizer_best.pt")
            torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/global_critic_optimizer_best.pt")  
        torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/global_actor_optimizer_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/global_critic_optimizer_periodic_{}.pt".format(step))
    
    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    if k == 'auc':
                        for i in range(self.max_episode_length):
                            wandb.log({k: np.mean(v[:,:,i])}, step=i+1)
                    else:
                        wandb.log({k: np.nanmean(v) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.95" else np.mean(v)}, step=total_num_steps)
                else:
                    if k == 'auc':
                        for i in range(self.max_episode_length):
                            self.writter.add_scalars(k, {k: np.mean(v[:,:,i])}, i+1)
                    else:
                        self.writter.add_scalars(k, {k: np.nanmean(v) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.95" else np.mean(v)}, total_num_steps)

    def log_agent(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if "merge" not in k:
                for agent_id in range(self.num_agents):
                    if k == "balanced_ratio":
                        for a in range(self.num_agents):
                            if a != agent_id:
                                agent_k = "agent{}/{}_".format(agent_id, a) + k
                    else:
                        agent_k = "agent{}_".format(agent_id) + k
                    if self.use_wandb:
                        wandb.log({agent_k: np.mean(np.array(v)[:,:,agent_id])}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(np.array(v)[:,:,agent_id])}, total_num_steps)
    
    def log_single_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if k in ['merge_explored_ratio_step', 'sum_merge_explored_ratio', 'merge_explored_ratio_step_0.95']:
                    for e in range(self.n_rollout_threads):
                        env_k = "{}_".format(self.scene_id[e])+k
                        if self.use_wandb:
                            wandb.log({env_k: np.nanmean(np.array(v)[:,e]) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.95" else np.mean(np.array(v)[:,e])}, step=total_num_steps)
                        else:
                            self.writter.add_scalars(env_k, {env_k: np.nanmean(np.array(v)[:,e]) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.95" else np.mean(np.array(v)[:,e])}, total_num_steps)
    
    def render_gifs(self):
        gif_dir = str(self.run_dir / 'gifs')

        folders = []
        get_folders(gif_dir, folders)
        filer_folders = [folder for folder in folders if "all" in folder or "merge" in folder]

        for folder in filer_folders:
            image_names = sorted(os.listdir(folder))

            frames = []
            for image_name in image_names:
                if image_name.split('.')[-1] == "gif":
                    continue
                image_path = os.path.join(folder, image_name)
                frame = imageio.imread(image_path)
                frames.append(frame)

            imageio.mimsave(str(folder) + '/render.gif', frames, duration=self.all_args.ifi)
    
    def visualize_obs(self, fig, ax, obs):
        # individual
        for agent_id in range(self.num_agents * 2):
            sub_ax = ax[agent_id]
            for i in range(8):
                sub_ax[i].clear()
                sub_ax[i].set_yticks([])
                sub_ax[i].set_xticks([])
                sub_ax[i].set_yticklabels([])
                sub_ax[i].set_xticklabels([])
                if agent_id < self.num_agents:
                    sub_ax[i].imshow(obs["global_merge_obs"][0, agent_id,i])
                elif agent_id >= self.num_agents and i<4:
                    sub_ax[i].imshow(obs["global_obs"][0, agent_id-self.num_agents,i])
        plt.gcf().canvas.flush_events()
        # plt.pause(0.1)
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()
    
    @torch.no_grad()
    def eval(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            start = time.time()
            # store each episode ratio or reward
            self.init_env_info()
            self.env_step = 0
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
            self.sim_map_size = np.array([infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)])
            self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
            if self.action_mask:
                self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute = True
            self.first_compute_global_input()

            # Compute Global goal
            rnn_states = self.eval_compute_global_goal(rnn_states)

            # compute local input

            if self.use_local_single_map:
                self.compute_local_input(self.single_merge_map)
            else:
                self.compute_local_input(self.merge_map)

            # Output stores local goals as well as the the ground-truth action
            self.global_output = self.envs.get_short_term_goal(self.global_insert)
            self.global_output = np.array(self.global_output, dtype = np.long)
            auc_area = np.zeros((self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)
            for step in range(self.max_episode_length):
                self.env_step = step
                if step % 15  == 0:
                    print(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio' and self.use_eval:
                                self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e][key])
                                auc_area[e, step] = auc_area[e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio' and self.use_eval:
                                self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e][key])
                                auc_single_area[e, :, step] = auc_single_area[e, :, step-1] + np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'explored_ratio_step':
                            for agent_id in range(self.num_agents):
                                agent_k = "agent{}_{}".format(agent_id, key)
                                if agent_k in infos[e].keys():
                                    self.env_info[key][e][agent_id] = infos[e][agent_k]
                        elif key == 'balanced_ratio':
                            for agent_id in range(self.num_agents):
                                for a in range(self.num_agents):
                                    agent_k = "agent{}/{}_{}".format(agent_id, a, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k] 
                        else:
                            if key in infos[e].keys():
                                self.env_info[key][e] = infos[e][key]
                    if self.env_info['sum_merge_explored_ratio'][e] <= self.all_args.explored_ratio_down_threshold:
                        self.env_info['merge_global_goal_num_%.2f'%self.all_args.explored_ratio_down_threshold][e] = self.env_info['merge_global_goal_num'][e]
                    
                    if self.num_agents==1:
                        if step in [49, 99, 149, 199, 249, 299, 349, 399, 449,359,719]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio']
                    else:
                        if step in [49, 99, 119, 149, 179, 199, 249, 299,209,89,599,479,419]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio']
                
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose(False)

                for a in range(self.num_agents):
                    if self.use_local_single_map:
                        self.single_merge_map[:, a] = self.single_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    else:
                        self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                

                # Global Policy
                if local_step == self.num_local_steps - 1:
                    # For every global step, update the full and local maps
                    self.update_map_and_pose()
                    self.compute_global_input()
                    self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    # Compute Global goal
                    rnn_states = self.eval_compute_global_goal(rnn_states)
                    self.env_info['merge_global_goal_num'] += self.num_agents
                
                    
                # Local Policy
                if self.use_local_single_map:
                    self.compute_local_input(self.single_merge_map)
                else:
                    self.compute_local_input(self.merge_map)

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.global_insert)
                self.global_output = np.array(self.global_output, dtype = np.long)
            
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            if not self.use_render :
                end = time.time()
                self.env_infos['merge_runtime'].append((end-start)/self.max_episode_length)
                self.log_env(self.env_infos, total_num_steps)
                self.log_single_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

        if self.all_args.save_gifs:
            print("generating gifs....")
            self.render_gifs()
            print("done!")
    
    @torch.no_grad()
    def eval_ft(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            start = time.time()
            # store each episode ratio or reward
            self.init_env_info()
            self.env_step = 0
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
            self.sim_map_size = np.array([infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)])
            self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
            if self.action_mask:
                self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute = True
            self.first_compute_global_input()

            # Compute Global goal
            for e in range(self.n_rollout_threads):
                self.ft_compute_global_goal(e)

            # compute local input

            self.ft_compute_local_input()

            # Output stores local goals as well as the the ground-truth action
            self.global_output = self.envs.get_short_term_goal(self.local_input)
            self.global_output = np.array(self.global_output, dtype = np.long)
            auc_area = np.zeros((self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)
            for step in range(self.max_episode_length):
                self.env_step = step
                if step % 15 == 0:
                    print(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio' and self.use_eval:
                                self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e][key])
                                auc_area[e, step] = auc_area[e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio' and self.use_eval:
                                self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e][key])
                                auc_single_area[e, :, step] = auc_single_area[e, :, step-1] + np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'explored_ratio_step':
                            for agent_id in range(self.num_agents):
                                agent_k = "agent{}_{}".format(agent_id, key)
                                if agent_k in infos[e].keys():
                                    self.env_info[key][e][agent_id] = infos[e][agent_k]
                        elif key == 'balanced_ratio':
                            for agent_id in range(self.num_agents):
                                for a in range(self.num_agents):
                                    agent_k = "agent{}/{}_{}".format(agent_id, a, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k] 
                        else:
                            if key in infos[e].keys():
                                self.env_info[key][e] = infos[e][key]
                    if self.env_info['sum_merge_explored_ratio'][e] <= self.all_args.explored_ratio_down_threshold:
                        self.env_info['merge_global_goal_num_%.2f'%self.all_args.explored_ratio_down_threshold][e] = self.env_info['merge_global_goal_num'][e]
                    
                    if self.num_agents==1:
                        if step in [49, 99, 149, 199, 249, 299, 349, 399, 449,359,719]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio']
                    else:
                        if step in [49, 99, 119, 149, 179, 199, 249, 299,209,89,599,479,419]:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                            self.env_info[str(step+1)+'step_merge_overlap_ratio'][e] = infos[e]['overlap_ratio'] 
                
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose()

                for a in range(self.num_agents):
                    self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    
                # Global Policy
                self.ft_go_steps += 1

                for e in range (self.n_rollout_threads):
                    self.ft_last_merge_explored_ratio[e] = self.env_info['sum_merge_explored_ratio'][e]
                    self.ft_compute_global_goal(e) 
                        
                # Local Policy
                self.ft_compute_local_input()

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.local_input)
                self.global_output = np.array(self.global_output, dtype = np.long)
            
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            if not self.use_render :
                end = time.time()
                self.env_infos['merge_runtime'].append((end-start)/self.max_episode_length)
                self.log_env(self.env_infos, total_num_steps)
                self.log_single_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

        if self.all_args.save_gifs:
            print("generating gifs....")
            self.render_gifs()
            print("done!")
            
            
    @torch.no_grad()
    def eval_auc(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            start = time.time()
            # store each episode ratio or reward
            self.init_env_info()
            self.env_step = 0
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
            self.sim_map_size = np.array([infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)])
            self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
            if self.action_mask:
                self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute = True
            self.first_compute_global_input()

            # Compute Global goal
            rnn_states = self.eval_compute_global_goal(rnn_states)

            # compute local input

            if self.use_local_single_map:
                self.compute_local_input(self.single_merge_map)
            else:
                self.compute_local_input(self.merge_map)

            # Output stores local goals as well as the the ground-truth action
            self.global_output = self.envs.get_short_term_goal(self.global_insert)
            self.global_output = np.array(self.global_output, dtype = np.long)
            auc_area = np.zeros((self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)
            for step in range(self.max_episode_length):
                self.env_step = step
                print(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e]['merge_explored_ratio'])
                    self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e]['merge_explored_ratio'])
                
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose(False)

                for a in range(self.num_agents):
                    if self.use_local_single_map:
                        self.single_merge_map[:, a] = self.single_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    else:
                        self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                

                # Global Policy
                if local_step == self.num_local_steps - 1:
                    # For every global step, update the full and local maps
                    self.update_map_and_pose()
                    self.compute_global_input()
                    self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    # Compute Global goal
                    rnn_states = self.eval_compute_global_goal(rnn_states)
                    self.env_info['merge_global_goal_num'] += self.num_agents
                
                    
                # Local Policy
                if self.use_local_single_map:
                    self.compute_local_input(self.single_merge_map)
                else:
                    self.compute_local_input(self.merge_map)

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.global_insert)
                self.global_output = np.array(self.global_output, dtype = np.long)
            
        with open(self.run_dir + '/auc.json', 'w') as json_file:
            json.dump(self.auc_infos['merge_auc'].tolist(), json_file, ensure_ascii=False)
        with open(self.run_dir + '/agent_auc.json', 'w') as json_file:
            json.dump(self.auc_infos['agent_auc'].tolist(), json_file, ensure_ascii=False)
            
        for i in range(self.max_episode_length):
            if self.use_wandb:
                wandb.log({'merge_auc': np.mean(self.auc_infos['merge_auc'][:, :, i])}, step=i+1)
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_auc".format(agent_id)
                    wandb.log({agent_k: np.mean(np.array(self.auc_infos['agent_auc'])[:, :, agent_id, i])}, step=i+1)
            
    @torch.no_grad()
    def eval_ft_auc(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            start = time.time()
            # store each episode ratio or reward
            self.init_env_info()
            self.env_step = 0
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
            self.sim_map_size = np.array([infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)])
            self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
            if self.action_mask:
                self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute = True
            self.first_compute_global_input()

            # Compute Global goal
            for e in range(self.n_rollout_threads):
                self.ft_compute_global_goal(e)

            # compute local input

            self.ft_compute_local_input()

            # Output stores local goals as well as the the ground-truth action
            self.global_output = self.envs.get_short_term_goal(self.local_input)
            self.global_output = np.array(self.global_output, dtype = np.long)
            auc_area = np.zeros((self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)
            for step in range(self.max_episode_length):
                self.env_step = step
                print(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e]['merge_explored_ratio'])
                    self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e]['merge_explored_ratio'])          
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose()

                for a in range(self.num_agents):
                    self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    
                # Global Policy
                self.ft_go_steps += 1

                for e in range (self.n_rollout_threads):
                    self.ft_last_merge_explored_ratio[e] = self.env_info['sum_merge_explored_ratio'][e]
                    self.ft_compute_global_goal(e) 
                        
                # Local Policy
                self.ft_compute_local_input()

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.local_input)
                self.global_output = np.array(self.global_output, dtype = np.long)
            
        with open(self.run_dir + '/auc.json', 'w') as json_file:
            json.dump(self.auc_infos['merge_auc'].tolist(), json_file, ensure_ascii=False)
        with open(self.run_dir + '/agent_auc.json', 'w') as json_file:
            json.dump(self.auc_infos['agent_auc'].tolist(), json_file, ensure_ascii=False)
            
        for i in range(self.max_episode_length):
            if self.use_wandb:
                wandb.log({'merge_auc': np.mean(self.auc_infos['merge_auc'][:, :, i])}, step=i+1)
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_auc".format(agent_id)
                    wandb.log({agent_k: np.mean(np.array(self.auc_infos['agent_auc'])[:, :, agent_id, i])}, step=i+1)


    @torch.no_grad()
    def eval_async(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            # store each episode ratio or reward
            start = time.time()
            self.init_env_info()
            self.env_step = 0
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = np.array([infos[e]['explorable_map'] for e in range(self.n_rollout_threads)])
            self.sim_map_size = np.array([infos[e]['sim_map_size'] for e in range(self.n_rollout_threads)])
            self.empty_map = np.array([infos[e]['empty_map'] for e in range(self.n_rollout_threads)])
            if self.action_mask:
                self.global_input['action_mask_obs'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)
                self.global_input['action_mask'] = np.ones((self.n_rollout_threads, self.num_agents, 1, self.grid_size, self.grid_size), dtype=np.int32)

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute = True
            self.first_compute_global_input()

            # Compute Global goal
            rnn_states = self.eval_compute_global_goal(rnn_states)

            # compute local input
            if self.use_local_single_map:
                self.compute_local_input(self.single_merge_map)
            else:
                self.compute_local_input(self.merge_map)

            # Output stores local goals as well as the the ground-truth action
            self.global_output = self.envs.get_short_term_goal(self.global_insert)
            self.global_output = np.array(self.global_output, dtype = np.long)
            async_local_step = np.ones((self.n_rollout_threads, self.num_agents))*-1
            for step in range(self.max_episode_length):
                print("step {}".format(step))
                self.env_step = step
                async_local_step += 1

                self.last_obs = copy.deepcopy(self.obs)

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio' and self.use_eval:
                                self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio' and self.use_eval:
                                self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'explored_ratio_step':
                            for agent_id in range(self.num_agents):
                                agent_k = "agent{}_{}".format(agent_id, key)
                                if agent_k in infos[e].keys():
                                    self.env_info[key][e][agent_id] = infos[e][agent_k]
                        else:
                            if key in infos[e].keys():
                                self.env_info[key][e] = infos[e][key]
                    if self.env_info['sum_merge_explored_ratio'][e] <= self.all_args.explored_ratio_down_threshold:
                        self.env_info['merge_global_goal_num_%.2f'%self.all_args.explored_ratio_down_threshold][e] = self.env_info['merge_global_goal_num'][e]
                
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose(False)

                for a in range(self.num_agents):
                    if self.use_local_single_map:
                        self.merge_map[:, a] = self.single_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    else:
                        self.merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                # Global Policy
                
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        locs = self.local_pose[e, a] + self.origins[e, a]
                        r, c = locs[1], locs[0]
                        loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                        int(c * 100.0 / self.map_resolution)]
                        goal_x = self.global_goal[e, a, 0]*(self.sim_map_size[e][a][0]+10)+self.center_w-self.sim_map_size[e][a][0]//2-5 
                        goal_y = self.global_goal[e, a, 1]*(self.sim_map_size[e][a][1]+10)+self.center_h-self.sim_map_size[e][a][1]//2-5
                        dis = pu.get_l2_distance(loc_r, goal_x, loc_c, goal_y)
                        
                        if async_local_step[e, a] == self.num_local_steps - 1 or dis < 2:
                            # For every global step, update the full and local maps
                            self.update_agent_map_and_pose(e, a)
                            self.compute_global_input()
                            self.global_masks[e, a, 0] = 1 
                            # Compute Global goal
                            rnn_states = self.eval_compute_single_global_goal(e, a, rnn_states)
                            async_local_step[e, a] = -1
                            self.env_info['merge_global_goal_num'] += 1
                
                    
                # Local Policy
                if self.use_local_single_map:
                    self.compute_local_input(self.single_merge_map)
                else:
                    self.compute_local_input(self.merge_map)

                # Output stores local goals as well as the the ground-truth action
                self.global_output = self.envs.get_short_term_goal(self.global_insert)
                self.global_output = np.array(self.global_output, dtype = np.long)
            
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            if not self.use_render :
                end = time.time()
                self.env_infos['merge_runtime'].append((end-start)/self.max_episode_length)
                self.log_env(self.env_infos, total_num_steps)
                self.log_single_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

        if self.all_args.save_gifs:
            print("generating gifs....")
            self.render_gifs()
            print("done!")