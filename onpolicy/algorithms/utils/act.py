
from .distributions import Bernoulli, Categorical, DiagGaussian
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args = None, device=torch.device("cpu")):
        super(ACTLayer, self).__init__()
        self.multidiscrete_action = False
        self.continuous_action = False
        self.mixed_action = False
        self.grid_goal = False
        self.grid_goal_simpler = False
        self.use_grid_simple = args.use_grid_simple
        self.action_mask = args.action_mask

        if args is not None and args.grid_goal:
            self.grid_goal = True
            if args.use_grid_simple:
                if self.action_mask:
                    self.to_input = Rearrange('b c h w -> b c (h w)', c = 3, h = args.grid_size, w = args.grid_size)
                else:
                    self.to_input = Rearrange('b c h w -> b c (h w)', c = 2, h = args.grid_size, w = args.grid_size)
            else:
                if self.action_mask:
                    self.to_input = Rearrange('b c h w -> b c (h w)', c = 4, h = args.grid_size, w = args.grid_size)
                else:
                    self.to_input = Rearrange('b c h w -> b c (h w)', c = 3, h = args.grid_size, w = args.grid_size)
            self.to_region = Categorical(args.grid_size ** 2, args.grid_size ** 2, use_orthogonal, gain)
            self.to_point = DiagGaussian(args.grid_size ** 2, 2, use_orthogonal, gain)
        elif args is not None and args.grid_goal_simpler:
            self.grid_goal_simpler = True
            if self.action_mask:
                self.to_input = Rearrange('b c h w -> b c (h w)', c = 2, h = args.grid_size, w = args.grid_size)
            else:
                self.to_input = Rearrange('b c h w -> b c (h w)', c = 1, h = args.grid_size, w = args.grid_size)
            self.to_region = Categorical(args.grid_size ** 2, args.grid_size ** 2, use_orthogonal, gain)
                
        elif action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                inputs_dim, discrete_dim, use_orthogonal, gain)])
    
    def forward(self, x, available_actions=None, deterministic=False):
        if self.grid_goal or self.grid_goal_simpler:
            x = self.to_input(x)

            action_log_probs = []
            # region
            if self.action_mask:
                region_logit = self.to_region(x[:, 0, :], x[:, -1, :], trans=False)
            else:
                region_logit = self.to_region(x[:, 0, :], trans=False)
            region = region_logit.mode() if deterministic else region_logit.sample()
            region_log_prob = region_logit.log_probs(region)
            action_log_probs.append(region_log_prob)
            
            # point
            if not self.grid_goal_simpler:
                if self.use_grid_simple:
                    point_logit = self.to_point(x[:, 1, :], trans=True)
                else:
                    regionk = []
                    for k, reg in zip(x, region):
                        regionk.append(k[1:3, reg])
                    regionk = torch.stack(regionk).view(x.size(0), -1)
                    point_logit = self.to_point(regionk, trans=False)
                
                point = point_logit.mode() if deterministic else point_logit.sample()
                point_log_prob = point_logit.log_probs(point)
                action_log_probs.append(point_log_prob)

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            if self.grid_goal_simpler:
                actions = region
            else:
                actions = torch.cat([region.type(point.dtype), point], dim = 1)
        
        elif self.mixed_action :
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        
        elif self.continuous_action:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        if self.mixed_action or self.multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_probs = action_logits.probs
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        if self.grid_goal or self.grid_goal_simpler:
            x = self.to_input(x)
            regions = action[:, 0].unsqueeze(dim = 1).long()
            points = action[:, 1:]

            action_log_probs = [] 
            dist_entropy = []

            # region
            if self.action_mask:
                region_logits = self.to_region(x[:, 0, :], x[:, -1, :], trans=False)
            else:
                region_logits = self.to_region(x[:, 0, :], trans=False)
            region_log_prob = region_logits.log_probs(regions)
            action_log_probs.append(region_log_prob)
            if active_masks is not None:
                if len(region_logits.entropy().shape) == len(active_masks.shape):
                    dist_entropy.append((region_logits.entropy() * active_masks).sum()/active_masks.sum()) 
                else:
                    dist_entropy.append((region_logits.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
            else:
                dist_entropy.append(region_logits.entropy().mean())

            # point
            if not self.grid_goal_simpler:
                if self.use_grid_simple:
                    point_logits = self.to_point(x[:, 1, :], trans=True)
                else:
                    regionk = []
                    for k, region in zip(x, regions):
                        regionk.append(k[1:3, region])
                    regionk = torch.stack(regionk).view(x.size(0), -1)
                    point_logits = self.to_point(regionk, trans=False)
                    
                point_log_prob = point_logits.log_probs(points)
                action_log_probs.append(point_log_prob)

                if active_masks is not None:
                    if len(point_logits.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((point_logits.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((point_logits.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(point_logits.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            if self.grid_goal_simpler:
                dist_entropy = dist_entropy[0]
            else:
                dist_entropy = dist_entropy[0] * 0.5 + dist_entropy[1] * 0.5
        
        elif self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] * 0.0025 + dist_entropy[1] * 0.01 

        elif self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        elif self.continuous_action:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()       
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy