import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .util import init
from .resnet import MapNet
from onpolicy.algorithms.utils.vit import ViT, Attention, PreNorm, Transformer, CrossAttention, FeedForward
from onpolicy.algorithms.utils.invariant import Invariant, AgentEncoder, AlterEncoder
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class MIXBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None, cnn_last_linear = True):
        super(MIXBase, self).__init__()

        self._agent_invariant = args.agent_invariant
        self.invariant_type = args.invariant_type
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self._multi_layer_cross_attn = args.multi_layer_cross_attn
        self._norm_sum = args.norm_sum
        self.hidden_size = args.hidden_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.use_resnet = args.use_resnet
        self.use_vit = args.use_vit
        self.use_grid_simple = args.use_grid_simple
        self.cnn_last_linear = cnn_last_linear
        self.grid_goal = args.grid_goal
        self.grid_size = args.grid_size
        self.cnn_use_attn = args.cnn_use_attn
        self.cnn_use_transformer = args.cnn_use_transformer
        self.action_mask = args.action_mask
        self.grid_agent_id = args.grid_agent_id
        self.grid_pos = args.grid_pos
        self.grid_last_goal = args.grid_last_goal
        self.use_self_grid_pos = args.use_self_grid_pos
        self.grid_goal_simpler = args.grid_goal_simpler
        self.use_enhanced_id = args.use_enhanced_id
        self.use_one_cnn_model = args.use_one_cnn_model
        self.use_share_cnn_model = args.use_share_cnn_model
        self.num_agents = args.num_agents
        self.use_explicit_id = args.use_explicit_id 
        self.use_id_embedding = args.use_id_embedding   
        self.use_pos_embedding = args.use_pos_embedding   
        self.use_intra_attn = args.use_intra_attn
        self.use_self_attn = args.use_self_attn
        self.depth = args.attn_depth
        self.add_grid_pos = args.add_grid_pos
        self.use_grid_pos_attn = args.use_grid_pos_attn
        self._take_self = args.take_self 
        self.use_flatten_attn = args.use_flatten_attn
        self.cnn_keys = []
        self.embed_keys = []
        self.mlp_keys = []
        self.local_cnn_keys = []
        self.transformer_keys = []
        self.n_cnn_input = 0
        self.n_embed_input = 0
        self.n_mlp_input = 0
        self.prev_out_channels = 0

        for key in obs_shape:
            if obs_shape[key].__class__.__name__ == 'Box':
                key_obs_shape = obs_shape[key].shape
                if len(key_obs_shape) == 3:
                    if key in ["local_obs", "local_merge_obs"]:
                        self.local_cnn_keys.append(key)
                    elif key in ["grid_pos",'grid_goal']:
                        self.transformer_keys.append(key)
                    elif key in ["grid_agent_id",'action_mask', 'action_mask_obs']:
                        pass
                    else:
                        self.cnn_keys.append(key)
                else:
                    if "orientation" in key:
                        self.embed_keys.append(key)
                    else:
                        self.mlp_keys.append(key)
            else:
                raise NotImplementedError

        if len(self.cnn_keys) > 0:
            if self.use_resnet:
                self.cnn = self._build_resnet_model(obs_shape, self.cnn_keys, self.hidden_size)
            elif self.use_vit:
                if self.use_one_cnn_model:
                        self.cnn = self._build_vit_model(obs_shape, self.cnn_keys, self.hidden_size, self._use_orthogonal, self._activation_id)
                else:
                    if self.use_share_cnn_model:
                        self.cnn = self._build_vit_model(obs_shape, self.cnn_keys, self.hidden_size, self._use_orthogonal, self._activation_id)
                    else:
                        for i in range(self.num_agents):
                            setattr(self, 'cnn_' + str(i), self._build_vit_model(obs_shape, self.cnn_keys, self.hidden_size, self._use_orthogonal, self._activation_id))
            else:
                if self.use_enhanced_id:
                    self.cnn_0 = self._build_first_cnn_model(obs_shape, self.cnn_keys, self.hidden_size, self._use_orthogonal, self._activation_id, cnn_layers_params = [(32,3,1,1)])
                    self.cnn_1 = self._build_middle_cnn_model(self.hidden_size, self._use_orthogonal, self._activation_id, cnn_layers_params = [(64,3,1,1)])
                    self.cnn_2 = self._build_middle_cnn_model(self.hidden_size, self._use_orthogonal, self._activation_id, cnn_layers_params = [(128,3,1,1)])
                    self.cnn_3 = self._build_middle_cnn_model(self.hidden_size, self._use_orthogonal, self._activation_id, cnn_layers_params = [(64,3,1,1)])
                    self.cnn_4 = self._build_middle_cnn_model(self.hidden_size, self._use_orthogonal, self._activation_id, cnn_layers_params = [(32,3,2,1)], use_maxpool2d=False)
                else:
                    if self.use_one_cnn_model:
                        self.cnn = self._build_cnn_model(obs_shape, self.cnn_keys, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._activation_id)
                    else:
                        if self.use_share_cnn_model:
                            self.cnn = self._build_cnn_model(obs_shape, self.cnn_keys, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._activation_id)
                        else:
                            for i in range(self.num_agents):
                                setattr(self, 'cnn_' + str(i), self._build_cnn_model(obs_shape, self.cnn_keys, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._activation_id))

        if self._agent_invariant and self.invariant_type not in ["split_attn", "alter_attn", "hama"]:
            self.invariant = self._build_invariant_model()

        if len(self.transformer_keys) >0:
            self.transformer = self._build_transformer_model(self._use_orthogonal, self._activation_id)
        
        if len(self.local_cnn_keys) > 0:
            if self.use_resnet:
                self.local_cnn = self._build_resnet_model(obs_shape, self.local_cnn_keys, self.hidden_size)
            else:
                self.local_cnn = self._build_cnn_model(obs_shape, self.local_cnn_keys, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._activation_id, True)
        
        if len(self.embed_keys) > 0:
            self.embed = self._build_embed_model(obs_shape)
        
        if len(self.mlp_keys) > 0:
            self.mlp = self._build_mlp_model(obs_shape, self.mlp_hidden_size, self._use_orthogonal, self._activation_id)
        
        if self._agent_invariant and self.invariant_type in ["split_attn", "alter_attn", "hama"]:
            self.descon = self._build_invariant_descon_model(self.hidden_size, self._use_orthogonal, self._activation_id)
        else:
            self.descon = self._build_descon_model(self.hidden_size, self._use_orthogonal, self._activation_id)
            
        if self._agent_invariant and self.use_grid_pos_attn:
            self.descon_again = self._build_invariant_descon_grid_pos_model(self.hidden_size, self._use_orthogonal, self._activation_id)
        
        print("descon", self.descon[0])

    def forward(self, x, dgn_analyze=False):
        out_x = x
        if len(self.cnn_keys) > 0:
            cnn_input = self._build_cnn_input(x, self.cnn_keys)   
            if self.use_enhanced_id:
                cnn_x = self.cnn_0(cnn_input)
                cnn_x = torch.cat([cnn_x, x['enhanced_id_1']], dim=1) 
                cnn_x = self.cnn_1(cnn_x)
                cnn_x = torch.cat([cnn_x, x['enhanced_id_2']], dim=1)
                cnn_x = self.cnn_2(cnn_x)
                cnn_x = torch.cat([cnn_x, x['enhanced_id_3']], dim=1)
                cnn_x = self.cnn_3(cnn_x)
                cnn_x = torch.cat([cnn_x, x['enhanced_id_4']], dim=1)
                out_x = self.cnn_4(cnn_x)
            else: 
                if self.use_one_cnn_model:
                    out_x = self.cnn(cnn_input)
                else:  
                    if self.use_share_cnn_model:
                        cnn_x = []
                        for i in range(self.num_agents):
                            cnn_x.append(self.cnn(cnn_input[:, 0+self.split_channel*i:self.split_channel+self.split_channel*i]))
                        out_x = torch.cat(cnn_x, dim=1)
                    else:
                        cnn_x = []
                        for i in range(self.num_agents):
                            exec('cnn_x.append(self.cnn_{}(cnn_input[:, 0+self.split_channel*i:self.split_channel+self.split_channel*i]))'.format(i))
                        out_x = torch.cat(cnn_x, dim=1)
        
        if self._agent_invariant:
            channels = out_x.shape[1] // self.num_agents
            actor_feature = out_x[:, 0:channels].clone()
            actor = rearrange(out_x[:, 0:channels], 'b c h w -> b (h w) c')
            others = [rearrange(out_x[:, i*channels:(i+1)*channels], 'b c h w ->  b (h w) c') for i in range(1, self.num_agents)]
            if len(self.transformer_keys)>0:
                actor_input, other_input = self._build_invariant_input(x, self.transformer_keys)
                actor_input = self.transformer(rearrange(actor_input, 'b c h w -> b (h w) c'))
                actor_trans_x = rearrange(actor_input, 'b (h w) c -> b c h w', h = self.grid_size)
                other_input = [self.transformer(rearrange(tmp, 'b c h w -> b (h w) c')) for tmp in other_input]
                if self.add_grid_pos:
                    actor = torch.cat([actor, actor_input], dim = -1)
                    others = [torch.cat([others[i], other_input[i]], dim = -1) for i in range(len(others))]
            if self.invariant_type in ["attn_rnn", "attn_sum", "mean", "attn_N"]:
                out_x = self.invariant(actor, others)
                out_x = rearrange(out_x, 'b (h w) c -> b c h w', h = self.grid_size, w = self.grid_size)
                out_x = torch.cat([out_x, actor_feature], dim=1)
                if len(self.transformer_keys)>0:
                    out_x = torch.cat([out_x, actor_trans_x], dim = 1)
            elif self.invariant_type in ["split_attn", "alter_attn", "hama"]:
                out_x = [actor, others]
            else:
                raise NotImplementedError
        else:
            if len(self.transformer_keys)>0:    
                transformer_input = self._build_transformer_input(x, self.transformer_keys)    
                trans_x = self.transformer(transformer_input)
                trans_shape = trans_x.shape
                trans_x = trans_x.permute(0,2,1).reshape(trans_shape[0], trans_shape[2], self.grid_size, self.grid_size)
                
                out_x = torch.cat([out_x, trans_x], dim=1)
            
            if self.grid_agent_id:
                out_x = torch.cat([out_x, x['grid_agent_id']], dim=1) 
            
            if self.action_mask:
                out_x = torch.cat([out_x, x['action_mask_obs']], dim=1) 


        if dgn_analyze:
            out_x, attn_weight = self.descon[0](out_x, dgn_analyze=True)
            return attn_weight
        else:
            out_x = self.descon(out_x)
        if (not self.cnn_last_linear) and self.use_grid_pos_attn:
            out_x  = torch.cat([out_x, actor_trans_x], dim=1) 
            out_x = self.descon_again(out_x)
            
        if len(self.local_cnn_keys) > 0:
            local_cnn_input = self._build_cnn_input(x, self.local_cnn_keys)
            local_cnn_x = self.local_cnn(local_cnn_input)            
            out_x = torch.cat([out_x, local_cnn_x], dim=1)

        if len(self.embed_keys) > 0:
            embed_input = self._build_embed_input(x)
            embed_x = (self.embed(embed_input.long()).view(embed_input.size(0), -1))            
            out_x = torch.cat([out_x, embed_x], dim=1)

        if len(self.mlp_keys) > 0:
            mlp_input = self._build_mlp_input(x)
            mlp_x = self.mlp(mlp_input).view(mlp_input.size(0), -1)
            out_x = torch.cat([out_x, mlp_x], dim=1) # ! wrong

        if self.action_mask and len(out_x.shape) > 2:
            out_x = torch.cat([out_x, x['action_mask']], dim=1)

        return out_x

    def _build_cnn_model(self, obs_shape, cnn_keys, cnn_layers_params, hidden_size, use_orthogonal, activation_id, local_cnn=False):
        
        if cnn_layers_params is None:
            cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        else:
            def _convert(params):
                output = []
                for l in params.split(' '):
                    output.append(tuple(map(int, l.split(','))))
                return output
            cnn_layers_params = _convert(cnn_layers_params)
        
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        n_cnn_input = 0
        for key in cnn_keys:
            if key in ['rgb','depth','image','occupy_image']:
                n_cnn_input += obs_shape[key].shape[2] 
                cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
            elif key in ['stack_obs']:
                if self.use_one_cnn_model:
                    n_cnn_input += obs_shape[key].shape[0]
                else:
                    n_cnn_input += int(obs_shape[key].shape[0] // self.num_agents)
                    self.split_channel = int(obs_shape[key].shape[0] // self.num_agents)
                cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
            else:
                pass

        cnn_layers = []
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_layers.append(nn.MaxPool2d(2))

            if i == 0:
                in_channels = n_cnn_input
            else:
                in_channels = self.prev_out_channels

            cnn_layers.append(init_(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
            cnn_layers.append(active_func)
            self.prev_out_channels = out_channels

        for i, (_, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_dims = self._maxpool_output_dim(dimension=cnn_dims,
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([2, 2], dtype=np.float32),
                stride=np.array([2, 2], dtype=np.float32))
            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )
        if local_cnn:
            if self.cnn_use_attn:
                print("cnn use attention")
                cnn_layers += [
                    Rearrange('b c h w -> b (h w) c', c = self.prev_out_channels, h = cnn_dims[0], w = cnn_dims[1]),
                    Attention(self.prev_out_channels, heads=4, dim_head=32, dropout=0.0),
                    Rearrange('b (h w) c -> b c h w', c = self.prev_out_channels, h = cnn_dims[0], w = cnn_dims[1])
                ]
            if self.cnn_use_transformer:
                print("cnn use transformer")
                cnn_layers += [
                    Rearrange('b c h w -> b (h w) c', c = self.prev_out_channels, h = cnn_dims[0], w = cnn_dims[1]),
                    nn.Linear(self.prev_out_channels, 128),
                    Transformer(dim=128, depth=2, heads=4, dim_head=32, mlp_dim=256),
                    PreNorm(128, nn.Linear(128,self.prev_out_channels)),
                    Rearrange('b (h w) c -> b c h w', c = self.prev_out_channels, h = cnn_dims[0], w = cnn_dims[1])
                ]

            if self.cnn_last_linear:
                cnn_layers += [
                    Flatten(),
                    init_(nn.Linear(self.prev_out_channels * cnn_dims[0] * cnn_dims[1], 
                                    hidden_size)),
                    active_func,
                    nn.LayerNorm(hidden_size),
                ]
                self.cnn_output_dim = hidden_size
            else:
                if self.use_grid_simple:
                    cnn_layers += [init_(nn.Conv2d(in_channels = self.prev_out_channels, out_channels = 2, kernel_size = 1, stride = 1, padding = 0))]
                    self.cnn_output_dim = 2 * cnn_dims[0] * cnn_dims[1]
                else:
                    cnn_layers += [init_(nn.Conv2d(in_channels = self.prev_out_channels, out_channels = 3, kernel_size = 1, stride = 1, padding = 0))]
                    self.cnn_output_dim = 3 * cnn_dims[0] * cnn_dims[1]
            
        self.cnn_dims = cnn_dims

        return nn.Sequential(*cnn_layers)

    def _build_first_cnn_model(self, obs_shape, cnn_keys, hidden_size, use_orthogonal, activation_id, cnn_layers_params = [(32, 3, 1, 1)]):

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        n_cnn_input = 0
        for key in cnn_keys:
            if key in ['rgb','depth','image','occupy_image']:
                n_cnn_input += obs_shape[key].shape[2] 
                self.cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
            elif key in ['global_map','local_map','global_obs','local_obs','global_merge_obs','local_merge_obs','trace_image', 'global_goal', 'global_merge_goal','gt_map', 'vector_cnn', 'direction_map']:
                n_cnn_input += obs_shape[key].shape[0] 
                self.cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
            else:
                raise NotImplementedError

        cnn_layers = []
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d:
                cnn_layers.append(nn.MaxPool2d(2))
                self.cnn_dims = self._maxpool_output_dim(dimension=self.cnn_dims,
                                                        dilation=np.array([1, 1], dtype=np.float32),
                                                        kernel_size=np.array([2, 2], dtype=np.float32),
                                                        stride=np.array([2, 2], dtype=np.float32))

            cnn_layers.append(init_(nn.Conv2d(in_channels=n_cnn_input,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
            cnn_layers.append(active_func)
            self.cnn_dims = self._cnn_output_dim(
                dimension=self.cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )
            self.prev_out_channels = out_channels

        return nn.Sequential(*cnn_layers)

    def _build_middle_cnn_model(self, hidden_size, use_orthogonal, activation_id, cnn_layers_params, use_maxpool2d=True):
        
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        cnn_layers = []
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and use_maxpool2d:
                cnn_layers.append(nn.MaxPool2d(2))
                self.cnn_dims = self._maxpool_output_dim(dimension=self.cnn_dims,
                                                        dilation=np.array([1, 1], dtype=np.float32),
                                                        kernel_size=np.array([2, 2], dtype=np.float32),
                                                        stride=np.array([2, 2], dtype=np.float32))
     
            in_channels = self.prev_out_channels
            # if self.use_enhanced_id:
            #     in_channels += 3 #obs_shape['enhanced_id'].shape[0]

            cnn_layers.append(init_(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
            cnn_layers.append(active_func)
            self.cnn_dims = self._cnn_output_dim(
                dimension=self.cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )
            self.prev_out_channels = out_channels

        return nn.Sequential(*cnn_layers)

    def _build_invariant_model(self):
        prev_channels = self.prev_out_channels
        in_channels = self.prev_out_channels
        if self.add_grid_pos:
            in_channels += 16
        if self.invariant_type == "mean":
            self.prev_out_channels = in_channels + prev_channels
        elif self.invariant_type in ["attn_sum", "attn_rnn", "attn_N"]:
            self.prev_out_channels = 64 + prev_channels
        else:
            raise NotImplementedError
        return Invariant(self.grid_size ** 2, in_channels, invariant_type = self.invariant_type, hidden_dim = 64, heads = 4, dim_head = 16, mlp_dim = 64)

    def _build_invariant_descon_model(self, hidden_size, use_orthogonal, activation_id):
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        in_channels = self.prev_out_channels
        if self.add_grid_pos:
            in_channels += 16
        
        if self.invariant_type == "split_attn":
            layers = [AgentEncoder(self.grid_size ** 2, in_channels, depth=2, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, pool = "one"),]
        elif self.invariant_type == "alter_attn":
            layers = [AlterEncoder(self.grid_size ** 2, in_channels, use_explicit_id = self.use_explicit_id, use_id_embedding = self.use_id_embedding, use_pos_embedding = self.use_pos_embedding, depth=self.depth, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, _multi_layer_cross_attn = self._multi_layer_cross_attn, _norm_sum = self._norm_sum, _take_self = self._take_self, use_flatten_attn = self.use_flatten_attn,\
                use_intra_attn = self.use_intra_attn, use_self_attn = self.use_self_attn)]
        elif self.invariant_type == "hama":
            layers = [AlterEncoder(self.grid_size ** 2, in_channels, use_explicit_id = self.use_explicit_id, use_id_embedding = self.use_id_embedding, use_pos_embedding = self.use_pos_embedding, depth=self.depth, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, _multi_layer_cross_attn = self._multi_layer_cross_attn, _norm_sum = self._norm_sum, _take_self = self._take_self, use_flatten_attn = self.use_flatten_attn,\
                hama=True)]
        else:
            raise NotImplementedError

        if self.cnn_last_linear:
            layers += [
                Flatten(),
                nn.Linear(128 * (self.grid_size ** 2), hidden_size),
                active_func,
                nn.LayerNorm(hidden_size)
            ]
        elif self.use_grid_pos_attn:
            layers += [ Rearrange('b (h w) d -> b d h w', h = self.grid_size) ]     
        else:
            layers += [
                nn.Linear(128, 2),
                Rearrange('b (h w) c -> b c h w', h = self.grid_size),
            ]

        return nn.Sequential(*layers)

    def _build_invariant_descon_grid_pos_model(self, hidden_size, use_orthogonal, activation_id):
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
                init_(nn.Conv2d(in_channels = 128 + 16, out_channels = 16, kernel_size = 1, stride = 1, padding = 0)), 
                active_func,

                Rearrange('b c h w -> b (h w) c', c = 16, h = 8, w = 8),
                Attention(16, heads=4, dim_head=32, dropout=0.0),
                Rearrange('b (h w) c -> b c h w', c = 16, h = 8, w = 8),

                init_(nn.Conv2d(in_channels = 16, out_channels = 2, kernel_size = 1, stride = 1, padding = 0))
            ]
        return nn.Sequential(*layers)
    
    def _build_descon_model(self, hidden_size, use_orthogonal, activation_id):

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        cnn_layers = []
        if not (self.use_one_cnn_model or self._agent_invariant):
            self.prev_out_channels = self.prev_out_channels * self.num_agents
        
        grid_c = 0

        if self.grid_pos or self.grid_last_goal:
            grid_c += 16
        if self.grid_agent_id:
            grid_c += 1
        if self.action_mask:
            grid_c += 1

        if grid_c != 0:  
            cnn_layers += [init_(nn.Conv2d(in_channels = self.prev_out_channels + grid_c, out_channels = 16, kernel_size = 1, stride = 1, padding = 0)), active_func]
        
        if self.cnn_use_attn:
            print("descon use attention")
            channels = 16 if grid_c!=0 else self.prev_out_channels
            cnn_layers += [
                Rearrange('b c h w -> b (h w) c', c = channels, h = self.cnn_dims[0], w = self.cnn_dims[1]),
                Attention(channels, heads=4, dim_head=32, dropout=0.0),
                Rearrange('b (h w) c -> b c h w', c = channels, h = self.cnn_dims[0], w = self.cnn_dims[1])
            ]
        if self.cnn_use_transformer:
            print("descon use transformer")
            channels = 16 if grid_c!=0 else self.prev_out_channels
            cnn_layers += [
                Rearrange('b c h w -> b (h w) c', c = channels, h = self.cnn_dims[0], w = self.cnn_dims[1]),
                nn.Linear(channels, 128),
                Transformer(dim=128, depth=2, heads=4, dim_head=32, mlp_dim=256),
                PreNorm(128, nn.Linear(128,channels)),
                Rearrange('b (h w) c -> b c h w', c = channels, h = self.cnn_dims[0], w = self.cnn_dims[1])
            ]

        if self.cnn_last_linear:
            if grid_c!=0:
                cnn_layers += [
                    Flatten(),
                    nn.Linear(16 * self.cnn_dims[0] * self.cnn_dims[1], 
                                    hidden_size),
                    active_func,
                    nn.LayerNorm(hidden_size),
                ]
            else:
                cnn_layers += [
                    Flatten(),
                    nn.Linear(self.prev_out_channels * self.cnn_dims[0] * self.cnn_dims[1], 
                                    hidden_size),
                    active_func,
                    nn.LayerNorm(hidden_size),
                ]
            
        else:
            if grid_c!=0: 
                if self.use_grid_simple:
                    cnn_layers += [init_(nn.Conv2d(in_channels = 16, out_channels = 2, kernel_size = 1, stride = 1, padding = 0))]
                    
                elif self.grid_goal_simpler:
                    cnn_layers += [init_(nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 1, stride = 1, padding = 0))]
                    
                else:
                    cnn_layers += [init_(nn.Conv2d(in_channels = 16, out_channels = 3, kernel_size = 1, stride = 1, padding = 0))]
            else:
                if self.use_grid_simple:
                    cnn_layers += [init_(nn.Conv2d(in_channels = self.prev_out_channels, out_channels = 2, kernel_size = 1, stride = 1, padding = 0))]
                elif self.grid_goal_simpler:
                    cnn_layers += [init_(nn.Conv2d(in_channels = self.prev_out_channels, out_channels = 1, kernel_size = 1, stride = 1, padding = 0))]                
                else:
                    cnn_layers += [init_(nn.Conv2d(in_channels = self.prev_out_channels, out_channels = 3, kernel_size = 1, stride = 1, padding = 0))]
                
        return nn.Sequential(*cnn_layers)

    def _build_resnet_model(self, obs_shape, cnn_keys, hidden_size):
        
        n_cnn_input = 0
        for key in cnn_keys:
            if key in ['rgb','depth','image','occupy_image']:
                n_cnn_input += obs_shape[key].shape[2] 
            elif key in ['global_map','local_map','global_obs','local_obs','global_merge_obs','local_merge_obs','trace_image', 'global_goal', 'global_merge_goal','gt_map','vector_cnn', 'direction_map']:
                n_cnn_input += obs_shape[key].shape[0] 
            else:
                raise NotImplementedError
            
        cnn_layers = [nn.Conv2d(int(n_cnn_input), 64, kernel_size=7, stride=2, padding=3, bias=False)]
        resnet = models.resnet18(pretrained = self.pretrained_global_resnet)
        cnn_layers += list(resnet.children())[1:-1]
        cnn_layers += [Flatten(),
                        nn.Linear(resnet.fc.in_features, hidden_size)]

        return nn.Sequential(*cnn_layers)
    
    def _build_vit_model(self, obs_shape, cnn_keys, hidden_size, use_orthogonal, activation_id):
        print("use vit")
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        n_cnn_input = 0
        for key in cnn_keys:
            if key in ['rgb','depth','image','occupy_image']:
                n_cnn_input += obs_shape[key].shape[2] 
                cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
            elif key in ['stack_obs']:
                if self.use_one_cnn_model:
                    n_cnn_input += obs_shape[key].shape[0]
                else:
                    n_cnn_input += int(obs_shape[key].shape[0] // self.num_agents)
                    self.split_channel = int(obs_shape[key].shape[0] // self.num_agents)
                cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
            else:
                pass
            
        vit = ViT(image_size = (int(cnn_dims[0]), int(cnn_dims[1])), patch_size = 16, num_classes = hidden_size, dim = 128, depth = 4, heads = 8, mlp_dim = 256, channels = n_cnn_input)

        models = [vit, init_(nn.Conv2d(in_channels=128,out_channels=32,kernel_size=3,stride=2,padding=1,)), 
                  active_func]
        
        self.cnn_dims = (int(self.grid_size), int(self.grid_size))
        self.prev_out_channels = 32
        # if not self.cnn_last_linear:
        #     if self.use_grid_simple:
        #         models += [nn.Linear(hidden_size, 2 * self.grid_size * self.grid_size),
        #             Rearrange('b (c h w) -> b c h w', c = 2, h = self.grid_size, w = self.grid_size)
        #         ]
        #     else:
        #         models += [nn.Linear(hidden_size, 3 * self.grid_size * self.grid_size),
        #             Rearrange('b (c h w) -> b c h w', c = 3, h = self.grid_size, w = self.grid_size)
        #         ]

        return nn.Sequential(*models)
    
    def _build_embed_model(self, obs_shape):
        self.embed_dim = 0
        for key in self.embed_keys:
            self.n_embed_input = 72
            self.n_embed_output = 8
            self.embed_dim += np.prod(obs_shape[key].shape)

        return nn.Embedding(self.n_embed_input, self.n_embed_output)
    
    def _build_transformer_model(self, use_orthogonal, activation_id,):
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)


        channels = 0
        if self._agent_invariant:
            if self.grid_pos:
                channels += 2
            if self.grid_last_goal:
                channels += 2
        else:
            if self.grid_pos:
                if self.use_self_grid_pos:
                    channels+=2
                else:
                    channels+=2*self.num_agents
            if self.grid_last_goal:
                if self.use_self_grid_pos:
                    channels+=2
                else:
                    channels+=2*self.num_agents
        
        cnn_layers = [init_(nn.Linear(channels,16)),
                    active_func,
                    nn.LayerNorm(16)
                ]
        return nn.Sequential(*cnn_layers)

    def _build_mlp_model(self, obs_shape, hidden_size, use_orthogonal, activation_id):

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for key in self.mlp_keys:
            self.n_mlp_input += np.prod(obs_shape[key].shape)

        return nn.Sequential(
                    init_(nn.Linear(self.n_mlp_input, hidden_size)), active_func, nn.LayerNorm(hidden_size))
         
    def _maxpool_output_dim(self, dimension, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    def _build_cnn_input(self, obs, cnn_keys):
        cnn_input = []

        for key in cnn_keys:
            if key in ['rgb','depth','image','occupy_image']:
                cnn_input.append(obs[key].permute(0, 3, 1, 2) / 255.0)
            elif key in ['stack_obs']:
                cnn_input.append(obs[key])
            else:
                pass

        cnn_input = torch.cat(cnn_input, dim=1)
        return cnn_input
    
    def _build_transformer_input(self, obs, transformer_keys):
        transformer_input = []
        for key in transformer_keys: 
            if key in ['grid_pos','grid_goal']:
                x = obs[key].shape
                transformer_input.append(obs[key].view(x[0],x[1],-1).permute(0,2,1))    
            else:
                raise NotImplementedError
        transformer_input = torch.cat(transformer_input, dim=2)
        return transformer_input

    def _build_invariant_input(self, obs, invariant_keys):
        invariant_input = {}
        for key in invariant_keys:
            if key in ['grid_pos', 'grid_goal']:
                invariant_input[key] = obs[key].chunk(self.num_agents, dim=1)
            else:
                raise NotImplementedError
        actor_input = []
        other_input = []
        for i in range(self.num_agents):
            tmp = torch.cat([invariant_input[key][i] for key in invariant_input.keys()],dim=1)
            if i == 0:
                actor_input = tmp.clone()
            else:
                other_input.append(tmp.clone())
        return actor_input, other_input

    def _build_embed_input(self, obs):
        embed_input = []
        for key in self.embed_keys:
            embed_input.append(obs[key].view(obs[key].size(0), -1))
        
        embed_input = torch.cat(embed_input, dim=1)
        return embed_input

    def _build_mlp_input(self, obs):
        mlp_input = []
        for key in self.mlp_keys:
            mlp_input.append(obs[key].view(obs[key].size(0), -1))

        mlp_input = torch.cat(mlp_input, dim=1)
        return mlp_input

    @property
    def output_size(self):
        output_size = 0
        if len(self.cnn_keys) > 0:
            output_size += self.hidden_size

        if len(self.embed_keys) > 0:
            output_size += 8 * self.embed_dim
        
        if len(self.local_cnn_keys) > 0:
            output_size += self.hidden_size

        if len(self.mlp_keys) > 0:
            output_size += self.mlp_hidden_size
        return output_size