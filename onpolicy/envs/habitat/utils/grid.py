import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=True)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=True)

    return rot_grid, trans_grid

def get_grid_full(pose, grid_size, full_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180. #pi
    
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=True)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=True)

    theta31 = torch.stack([cos_t, sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta32 = torch.stack([-sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta3 = torch.stack([theta31, theta32], 1)

    theta41 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), -x*grid_size[2]/full_size[2]], 1)
    theta42 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), -y*grid_size[3]/full_size[3]], 1)
    theta4 = torch.stack([theta41, theta42], 1)

    n_rot_grid = F.affine_grid(theta3, torch.Size(full_size), align_corners=True)
    
    n_trans_grid = F.affine_grid(theta4, torch.Size(full_size), align_corners=True)

    return rot_grid, trans_grid, n_rot_grid, n_trans_grid

def get_grid_reverse_full(pose, grid_size, full_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180. #pi
    
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=True)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=True)

    theta31 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta32 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta3 = torch.stack([theta31, theta32], 1)
    
    theta41 = torch.stack([torch.ones(x.shape).to(device),
                           torch.zeros(x.shape).to(device), x*grid_size[2]/full_size[2]], 1)
    theta42 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y*grid_size[3]/full_size[3]], 1)
    theta4 = torch.stack([theta41, theta42], 1)

    n_rot_grid = F.affine_grid(theta3, torch.Size(full_size), align_corners=True)
    
    n_trans_grid = F.affine_grid(theta4, torch.Size(full_size), align_corners=True)

    return rot_grid, trans_grid, n_rot_grid, n_trans_grid

def circular_area(size, s, r):
    H, W = size
    x, y = s
    a = np.zeros((H, W), dtype=np.int32)
    row = (np.arange(H)-x)**2
    row = row.repeat(W).reshape(H, W)
    col = (np.arange(W)-y)**2
    col = col.repeat(H).reshape(W, H).transpose()
    dist = (row+col)**0.5
    a[dist<=r] = 1
    return a
