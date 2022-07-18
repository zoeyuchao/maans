from onpolicy.envs.habitat.model.model import Local_IL_Policy
import numpy as np
import torch
from onpolicy.envs.habitat.utils.frontier import get_frontier, rrt_global_plan
import matplotlib
import matplotlib.pyplot as plt
import os

class goal_planner:
    def __init__(self, args, run_dir):
        self.all_args = args
        self.map_size_cm = args.map_size_cm
        self.map_resolution = args.map_resolution
        self.direction_k = args.direction_k
        self.direction_mode = args.direction_mode
        self.run_dir = str(run_dir)
        self.save_dir = os.path.join(self.run_dir, 'rrt')
        self.prepared = False
        if self.direction_mode == 'discrete':
            self.generate_gaps()
    
    def generate_gaps(self):
        unit = np.pi * 2 / self.direction_k
        width = unit
        self.sections = [(unit * i - width/2, unit * i +  width/2) for i in range(self.direction_k)]
    
    def prepare(self, world_merge_map, world_locs, n_rollout_threads, num_agents, env_step, envs = None, parallel = False):
        if self.direction_mode == 'continuous':
            return None
        self.prepared = False # True
        # goal_map, targets_map, success, goal_trace, rrt_trace, goals 
        self.direction_trace = np.zeros((n_rollout_threads, num_agents, 1, world_merge_map.shape[2], world_merge_map.shape[3]), dtype=np.float32) # rrt_trace
        self.direction_map = np.zeros((n_rollout_threads, num_agents, self.direction_k, 3, world_merge_map.shape[2], world_merge_map.shape[3]), dtype=np.float32) # 0 for goal_map, 1 for targets_map, 2 for goal_trace
        self.direction_vec = np.zeros((n_rollout_threads, num_agents, self.direction_k), dtype=np.float32) # success 
        self.direction_goal = np.zeros((n_rollout_threads, num_agents, self.direction_k, 2), dtype=np.float32) # goals

        if envs is None or parallel == False:
            raise NotImplementedError
        else:
            data = []
            for e in range(n_rollout_threads):
                x = {
                    'world_merge_map' : world_merge_map[e],
                    'world_locs' : world_locs[e],
                    'env_step' : env_step,
                    'sections': self.sections
                    }
                data.append(x)
            output = envs.prepare_direction_input(data)
            for e, (goal_map, targets_map, success, goal_trace, rrt_trace, goals) in enumerate(output):
                self.direction_map[e, :, :, 0] = goal_map
                self.direction_map[e, :, :, 1] = targets_map
                self.direction_trace[e, :, 0] = rrt_trace
                self.direction_map[e, :, :, 2] = goal_trace
                self.direction_vec[e] = success
                self.direction_goal[e] = goals
        direction_map = np.concatenate([self.direction_map.reshape(n_rollout_threads, num_agents, self.direction_k*3, world_merge_map.shape[2], world_merge_map.shape[3]), self.direction_trace], axis = 2)
        return direction_map, self.direction_vec
                    
    def get_goal(self, world_merge_map, world_locs, actions, n_rollout_threads, num_agents, env_step, envs = None, parallel = False):
        if envs is None or parallel == False:
            raise NotImplementedError
        else:
            goals = np.zeros((n_rollout_threads, num_agents, 2), dtype=np.float32)
            for e in range(n_rollout_threads):
                for a in range(num_agents):
                    k = int(actions[e][a])
                    goals[e, a] = self.direction_goal[e, a, k]
        return goals

    def greedy(self, global_input, n_rollout_threads, num_agents):
        '''
        a greedy action chooser
        '''
        actions = np.zeros((n_rollout_threads, num_agents), dtype=np.int32)
        threshold = 100.0
        area_threshold = 1000.0
        for e in range(n_rollout_threads):
            for a in range(num_agents):
                candidates = [k for k in range(self.direction_k) if self.direction_vec[e, a, k] == 1]
                print(e, a, candidates)
                if len(candidates) == 0:
                    candidates = list(range(self.direction_k))
                import random
                actions[e, a] = random.choice(candidates)
        return actions
                
