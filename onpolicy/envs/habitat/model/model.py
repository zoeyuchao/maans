import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from onpolicy.algorithms.utils.distributions import Categorical, DiagGaussian
from onpolicy.envs.habitat.utils.grid import get_grid
from onpolicy.algorithms.utils.util import init, check

def _t2n(x):
    return x.detach().cpu().numpy()

class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L82
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):

        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        masks = masks.squeeze(-1)
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs

# Neural SLAM Module code
class Neural_SLAM_Module(nn.Module):
    """
    """

    def __init__(self, args, device=torch.device("cuda:0")):
        super(Neural_SLAM_Module, self).__init__()

        self.device = device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.map_size_cm = int(args.map_size_cm // args.global_downscaling)
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.use_pe = args.use_pose_estimation
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Visual Encoding
        resnet = models.resnet18(pretrained=args.pretrained_resnet)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1,
                                 self.n_channels,
                                 self.screen_h,
                                 self.screen_w)
        conv_output = self.conv(self.resnet_l5(input_test))

        self.pool = ChannelPool(1)
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layer
        self.proj1 = nn.Linear(self.conv_output_size, 1024)
        self.proj2 = nn.Linear(1024, 4096)

        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
            self.dropout2 = nn.Dropout(self.dropout)

        # Deconv layers to predict map
        self.deconv = nn.Sequential(*filter(bool, [
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, (4, 4), stride=(2, 2), padding=(1, 1)),
        ]))

        # Pose Estimator
        self.pose_conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(4, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
            nn.ReLU()
        ]))

        pose_conv_output = self.pose_conv(torch.randn(1, 4,
                                                      self.vision_range,
                                                      self.vision_range))
        self.pose_conv_output_size = pose_conv_output.view(-1).size(0)

        # projection layer
        self.pose_proj1 = nn.Linear(self.pose_conv_output_size, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_y = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)

        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)

        self.st_poses_eval = torch.zeros(args.n_rollout_threads,
                                         3).to(device)
        self.st_poses_train = torch.zeros(args.slam_batch_size,
                                          3).to(device)

        grid_size = self.vision_range * 2
        self.grid_map_eval = torch.zeros(args.n_rollout_threads, 2,
                                         grid_size, grid_size
                                         ).float().to(device)
        self.grid_map_train = torch.zeros(args.slam_batch_size, 2,
                                          grid_size, grid_size
                                          ).float().to(device)

        self.agent_view = torch.zeros(args.n_rollout_threads, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(device)

        self.to(device)

    def forward(self, obs_last, obs, poses, maps, explored, current_poses, build_maps=True):

        obs = check(obs).to(**self.tpdv)
        obs_last = check(obs_last).to(**self.tpdv)
        poses = check(poses).to(**self.tpdv)
        if maps is not None:
            maps = check(maps).to(**self.tpdv)
        if explored is not None:
            explored = check(explored).to(**self.tpdv)
        if current_poses is not None:
            current_poses = check(current_poses).to(**self.tpdv)

        # Get egocentric map prediction for the current obs
        bs, c, h, w = obs.size()
        resnet_output = self.resnet_l5(obs[:, :3, :, :])
        conv_output = self.conv(resnet_output)

        proj1 = nn.ReLU()(self.proj1(
                          conv_output.view(-1, self.conv_output_size)))
        if self.dropout > 0:
            proj1 = self.dropout1(proj1)
        proj3 = nn.ReLU()(self.proj2(proj1))

        deconv_input = proj3.view(bs, 64, 8, 8)
        deconv_output = self.deconv(deconv_input)
        pred = torch.sigmoid(deconv_output)

        proj_pred = pred[:, :1, :, :]
        fp_exp_pred = pred[:, 1:, :, :]

        with torch.no_grad():
            # Get egocentric map prediction for the last obs
            bs, c, h, w = obs_last.size()
            resnet_output = self.resnet_l5(obs_last[:, :3, :, :])
            conv_output = self.conv(resnet_output)

            proj1 = nn.ReLU()(self.proj1(
                              conv_output.view(-1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)
            proj3 = nn.ReLU()(self.proj2(proj1))

            deconv_input = proj3.view(bs, 64, 8, 8)
            deconv_output = self.deconv(deconv_input)
            pred_last = torch.sigmoid(deconv_output)

            # ST of proj
            vr = self.vision_range
            grid_size = vr * 2

            if build_maps:
                st_poses = self.st_poses_eval.detach_()
                grid_map = self.grid_map_eval.detach_()
            else:
                st_poses = self.st_poses_train.detach_()
                grid_map = self.grid_map_train.detach_()

            st_poses.fill_(0.)
            st_poses[:, 0] = poses[:, 1] * 200. / self.resolution / grid_size
            st_poses[:, 1] = poses[:, 0] * 200. / self.resolution / grid_size
            st_poses[:, 2] = poses[:, 2] * 57.29577951308232
            rot_mat, trans_mat = get_grid(st_poses,
                                          (bs, 2, grid_size, grid_size),
                                          self.device)

            grid_map.fill_(0.)
            grid_map[:, :, vr:, int(vr / 2):int(vr / 2 + vr)] = pred_last
            translated = F.grid_sample(grid_map, trans_mat, align_corners=True)
            rotated = F.grid_sample(translated, rot_mat, align_corners=True)
            rotated = rotated[:, :, vr:, int(vr / 2):int(vr / 2 + vr)]

            pred_last_st = rotated

        # Pose estimator
        pose_est_input = torch.cat((pred.detach(), pred_last_st.detach()),
                                   dim=1)
        pose_conv_output = self.pose_conv(pose_est_input)
        pose_conv_output = pose_conv_output.view(-1,
                                                 self.pose_conv_output_size)

        proj1 = nn.ReLU()(self.pose_proj1(pose_conv_output))

        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)

        proj2_x = nn.ReLU()(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)

        proj2_y = nn.ReLU()(self.pose_proj2_y(proj1))
        pred_dy = self.pose_proj3_y(proj2_y)

        proj2_o = nn.ReLU()(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)

        pose_pred = torch.cat((pred_dx, pred_dy, pred_do), dim=1)
        if self.use_pe == 0:
            pose_pred = pose_pred * self.use_pe

        if build_maps:
            # Aggregate egocentric map prediction in the geocentric map
            # using the predicted pose
            with torch.no_grad():
                agent_view = self.agent_view.detach_()
                agent_view.fill_(0.)

                x1 = self.map_size_cm // (self.resolution * 2) \
                        - self.vision_range // 2
                x2 = x1 + self.vision_range
                y1 = self.map_size_cm // (self.resolution * 2)
                y2 = y1 + self.vision_range
                agent_view[:, :, y1:y2, x1:x2] = pred

                corrected_pose = poses + pose_pred

                def get_new_pose_batch(pose, rel_pose_change):
                    pose[:, 1] += rel_pose_change[:, 0] * \
                                  torch.sin(pose[:, 2] / 57.29577951308232) \
                                  + rel_pose_change[:, 1] * \
                                  torch.cos(pose[:, 2] / 57.29577951308232)
                    pose[:, 0] += rel_pose_change[:, 0] * \
                                  torch.cos(pose[:, 2] / 57.29577951308232) \
                                  - rel_pose_change[:, 1] * \
                                  torch.sin(pose[:, 2] / 57.29577951308232)
                    pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

                    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
                    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

                    return pose

                current_poses = get_new_pose_batch(current_poses,
                                                   corrected_pose)
                st_pose = current_poses.clone().detach()

                st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution
                                    - self.map_size_cm \
                                    // (self.resolution * 2)) \
                                 / (self.map_size_cm // (self.resolution * 2))
                st_pose[:, 2] = 90. - (st_pose[:, 2])

                rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                              self.device)

                rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
                translated = F.grid_sample(rotated, trans_mat, align_corners=True)

                maps2 = torch.cat((maps.unsqueeze(1),
                                   translated[:, :1, :, :]), 1)
                explored2 = torch.cat((explored.unsqueeze(1),
                                       translated[:, 1:, :, :]), 1)

                map_pred = self.pool(maps2).squeeze(1)
                exp_pred = self.pool(explored2).squeeze(1)

        else:
            map_pred = None
            exp_pred = None
            current_poses = None
        
        return proj_pred, fp_exp_pred, map_pred if map_pred == None else _t2n(map_pred),\
               exp_pred if exp_pred == None else _t2n(exp_pred), pose_pred,\
               current_poses if current_poses == None else _t2n(current_poses)


# Local Policy model code
class Local_IL_Policy(NNBase):

    def __init__(self, input_shape, num_actions, recurrent=False,
                 hidden_size=512, deterministic=False , device = torch.device("cuda:0")):

        super(Local_IL_Policy, self).__init__(recurrent, hidden_size, hidden_size)
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.deterministic = deterministic
        self.dropout = 0.5

        resnet = models.resnet18(pretrained=True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])

        # Extra convolution layer
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1, 3, input_shape[1], input_shape[2])
        conv_output = self.conv(self.resnet_l5(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        self.proj1 = nn.Linear(self.conv_output_size, hidden_size - 16)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)

        # Short-term goal embedding layers
        self.embedding_angle = nn.Embedding(72, 8)
        self.embedding_dist = nn.Embedding(24, 8)

        # Policy linear layer
        self.policy_linear = nn.Linear(hidden_size, num_actions)

        self.to(device)
        self.train()

    def forward(self, rgb, rnn_hxs, masks, extras):
        rgb = check(rgb).to(**self.tpdv)
        rnn_hxs = check(rnn_hxs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        extras = check(extras).to(**self.tpdv)

        if self.deterministic:
            x = torch.zeros(extras.size(0), 3)
            for i, stg in enumerate(extras):
                if stg[0] < 3 or stg[0] > 68:
                    x[i] = torch.tensor([0.0, 0.0, 1.0])
                elif stg[0] < 36:
                    x[i] = torch.tensor([0.0, 1.0, 0.0])
                else:
                    x[i] = torch.tensor([1.0, 0.0, 0.0])
        else:
            resnet_output = self.resnet_l5(rgb[:, :3, :, :])
            conv_output = self.conv(resnet_output)

            proj1 = nn.ReLU()(self.proj1(conv_output.view(
                -1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)

            angle_emb = self.embedding_angle(extras[:, 0].long()).view(-1, 8)
            dist_emb = self.embedding_dist(extras[:, 1].long()).view(-1, 8)
            x = torch.cat((proj1, angle_emb, dist_emb), 1)
            x = nn.ReLU()(self.linear(x))
            if self.is_recurrent:
                x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

            x = nn.Softmax(dim=1)(self.policy_linear(x))

        action = torch.argmax(x, dim=1)

        return action, x, rnn_hxs