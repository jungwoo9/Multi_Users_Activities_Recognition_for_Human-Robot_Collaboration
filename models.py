import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import math

import numpy as np
import scipy.sparse as sp

# define an LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, 150, num_layers, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(150)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(150, 9)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1).float()
        # print(x.shape) # torch.Size([256, 130, 60])
        lstm_out, _ = self.lstm(x)
        lstm_out = self.batch_norm1(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)
        # print(lstm_out.shape) # torch.Size([256, 130, 250])
        lstm_out2, _ = self.lstm2(lstm_out)
        lstm_out2 = self.batch_norm2(lstm_out2.permute(0, 2, 1)).permute(0, 2, 1)
        # print(lstm_out2.shape) # torch.Size([256, 130, 150])
        lstm_out2 = lstm_out2[:, -1, :]
        # print(lstm_out2.shape) # torch.Size([256, 150])
        hidden_output = self.fc1(lstm_out2)
        output_probs = F.softmax(hidden_output, dim=1)
        return output_probs

def get_lstm(input_size=3*20, hidden_size=250, output_size=9, device='cpu'):
    # Initialize the model
    lstm = LSTMModel(input_size, hidden_size, output_size, 1).to(device)

    return lstm
    
# define stgcn
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x

        return x

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)

        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0

        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result.to(input.dtype)

class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * residual connection *
    #        |                                |
    #        |    |--->--- casualconv2d ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- casualconv2d --- sigmoid ---|
    #

    #param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.act_func = act_func
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                # Explanation of Gated Linear Units (GLU):
                # The concept of GLU was first introduced in the paper
                # "Language Modeling with Gated Convolutional Networks".
                # URL: https://arxiv.org/abs/1612.08083
                # In the GLU operation, the input tensor X is divided into two tensors, X_a and X_b,
                # along a specific dimension.
                # In PyTorch, GLU is computed as the element-wise multiplication of X_a and sigmoid(X_b).
                # More information can be found here: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # The provided code snippet, (x_p + x_in) ⊙ sigmoid(x_q), is an example of GLU operation.
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))

            else:
                # tanh(x_p + x_in) ⊙ sigmoid(x_q)
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')

        return x

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        #bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv

class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        # if self.graph_conv_type == 'cheb_graph_conv':
        #     self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        if self.graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        # if self.graph_conv_type == 'cheb_graph_conv':
        #     x_gc = self.cheb_graph_conv(x_gc_in)
        if self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)

        return x_gc_out

class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())

    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id

    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso
    

# define encoder with stgcn
class Encoder1(torch.nn.Module):
  def __init__(self, args, device):
    super().__init__()

    self.device = device
    args.gso = args.gso.to(device)

    modules = []
    
    modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=3, channels=[64, 16, 64], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    # modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=64, channels=[64, 16, 64], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    # modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=64, channels=[64, 16, 64], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=64, channels=[64, 16, 32], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    modules.append(nn.AvgPool2d(kernel_size=(3, 3)))
    st_blocks = nn.Sequential(*modules)

    self.encoder = torch.nn.Sequential(
        st_blocks,
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=6144, # 32 * 22 * 6
                        out_features=2600),
        torch.nn.BatchNorm1d(2600),
        torch.nn.ReLU(),
    )
    
    # mean and sigma layer
    self.z_mean = torch.nn.Linear(in_features=2600,
                                  out_features=1950)
    self.z_log_var = torch.nn.Linear(in_features=2600,
                                     out_features=1950)

  def reparameterization(self, z_mean, z_log_var):
    std = torch.exp(z_log_var / 2)
    q = torch.distributions.Normal(z_mean, std)
    z = q.rsample()
    
    return z.to(self.device)

  def forward(self, x):
    x = x.permute(0,2,1,3).to(torch.float32)
    x = self.encoder(x)
    z_mean = self.z_mean(x)
    z_log_var = self.z_log_var(x)
    z = self.reparameterization(z_mean, z_log_var)

    return z, z_mean, z_log_var




## NEW STGCN ##########################

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, 3))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, A_hat, residual):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.dropout = nn.Dropout(0.3)
        self.reset_parameters()
        
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = lambda x: x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        A_hat = self.A_hat
        res = self.residual(X)
        t = self.temporal1(X)
        # t = self.batch_norm(t)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.batch_norm(self.temporal2(t2))
        
        out = t3 + res
        # out = t3
        
        return out
        # return t3

# define encoder with stgcn
class Encoder2(torch.nn.Module):
  def __init__(self, args, device):
    super().__init__()

    self.device = device
    args.gso = args.gso.to(device)

    modules = []
    
    modules.append(STGCNBlock(3, 16, 64, 20, args.gso, False))
    modules.append(STGCNBlock(64, 16, 64, 20, args.gso, True))
    modules.append(STGCNBlock(64, 16, 64, 20, args.gso, True))
    modules.append(STGCNBlock(64, 16, 64, 20, args.gso, True))
    modules.append(nn.AvgPool2d(kernel_size=(2, 2)))
    st_blocks = nn.Sequential(*modules)

    self.encoder = torch.nn.Sequential(
        st_blocks,
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=65*20*32, # 32 * 22 * 6
                        out_features=2600),
        # torch.nn.BatchNorm1d(2600),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features=2600,
                        out_features=3900),
        # torch.nn.BatchNorm1d(3900),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.3),
    )

  def reparameterization(self, z_mean, z_log_var):
    std = torch.exp(z_log_var / 2)
    q = torch.distributions.Normal(z_mean, std)
    z = q.rsample()
    
    return z.to(self.device)

  def forward(self, x):
      # 256 130 3 20
    x = x.permute(0,3,1,2).to(torch.float32)
    x = self.encoder(x)
    z_mean = x[:, :1950]
    z_log_var = x[:, 1950:]
    z = self.reparameterization(z_mean, z_log_var)

    return z, z_mean, z_log_var

#######################################



## Second version STGCN ###############

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2

        elif layout == 'custom':
          self.num_node = 20
          self_link = [(i, i) for i in range(self.num_node)]
          neighbor_link = [(0, 1), (1, 2), (1, 3), (1, 6), (2, 9), (3, 4), (4, 5), (6, 7), (7, 8),
                            (10, 11), (11, 12), (11, 13), (11, 16), (12, 19), (13, 14), (14, 15), (16, 17), (17, 18)]
          neighbor_link = neighbor_link + [(j, i) for (i, j) in neighbor_link]
          self.edge = self_link + neighbor_link
          self.center = 1
        
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

def get_graph():
    g = Graph('custom', 'spatial')
    A = torch.tensor(g.A, dtype=torch.float32, requires_grad=False)
    
    return A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 graph=None,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.graph = graph

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = self.graph

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)

class Encoder(torch.nn.Module):
  def __init__(self, args, device):
    super().__init__()

    self.device = device
    self.graph = args.gso.to(device)

    modules = []
    
    modules.append(st_gcn(3, 64, (9, 3), graph=self.graph, residual=False))
    modules.append(st_gcn(64, 64, (9, 3), graph=self.graph, residual=True))
    modules.append(st_gcn(64, 64, (9, 3), graph=self.graph, residual=True))
    modules.append(st_gcn(64, 64, (9, 3), graph=self.graph, residual=True))
    modules.append(nn.AvgPool2d(kernel_size=(2, 2)))
    st_blocks = nn.Sequential(*modules)

    self.encoder = torch.nn.Sequential(
        st_blocks,
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=65*20*32, # 32 * 22 * 6
                        out_features=2600),
        # torch.nn.BatchNorm1d(2600),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features=2600,
                        out_features=3900),
        # torch.nn.BatchNorm1d(3900),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.3),
    )

  def reparameterization(self, z_mean, z_log_var):
    std = torch.exp(z_log_var / 2)
    q = torch.distributions.Normal(z_mean, std)
    z = q.rsample()
    
    return z.to(self.device)

  def forward(self, x):
      # 256 130 3 20
    x = x.permute(0,2,1,3).to(torch.float32)
    x = self.encoder(x)
    z_mean = x[:, :1950]
    z_log_var = x[:, 1950:]
    z = self.reparameterization(z_mean, z_log_var)

    return z, z_mean, z_log_var

####################################### 

# define decoder
class Decoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.decoder_hidden = torch.nn.Sequential(
        torch.nn.Linear(in_features=1950,
                        out_features=3900),
        # torch.nn.BatchNorm1d(3900),
        # torch.nn.ReLU(),
        torch.nn.Linear(in_features=3900,
                        out_features=7800),
        # torch.nn.BatchNorm1d(7800),
        # torch.nn.ReLU(),
        torch.nn.Linear(in_features=7800,
                        out_features=15600),
        torch.nn.Tanh()
        # torch.nn.Sigmoid()
    )

  def forward(self, x):
    x = self.decoder_hidden(x)
    x = x.reshape(x.shape[0], 130, 6, 20)
    
    return x[:, :, :3, :], x[:, :, 3:, :]

# define vae
class VAE(torch.nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder =decoder

  def forward(self, x):
    z, z_mean, z_log_var = self.encoder(x)
    x_mean, x_var = self.decoder(z)
    return x_mean, x_var, z, z_mean, z_log_var

# define configurations for stgcn
class ModelConfig:
    def __init__(self):
        self.Kt = 9  # kernel for temporal convolution
        self.Ks = 3  # kernel for graph convolution
        self.n_his = 130  # number of historical time steps
        self.act_func = 'glu'
        self.n_vertex = 20 # number of vertex in data
        self.graph_conv_type = 'graph_conv'
        self.gso = get_graph()  # graph shift operator | adjacent matrix
        self.enable_bias = True  # enable bias terms in linear layers
        self.droprate = 0.3

def get_adj():
    adj = np.array([
        #0  2  3  5  6  8  12 13 15 26 0  2  3  5  6  8  12 13 15 26
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 8
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 12
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 13
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 15
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 26
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0], # 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], # 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], # 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], # 26
    ])
    adj = adj.astype(dtype=np.float32)
    adj = torch.from_numpy(adj)
    return adj

def get_gso():
    adj = np.array([
        #0  2  3  5  6  8  12 13 15 26 0  2  3  5  6  8  12 13 15 26
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 8
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 12
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 13
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 15
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 26
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0], # 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], # 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], # 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], # 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], # 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], # 26
    ])
    
    # Convert the NumPy array to a sparse matrix (Compressed Sparse Column format)
    dir_adj_sparse = sp.csc_matrix(adj)
    
    # Example graph shift operator type (choose one)
    gso_type = 'sym_norm_adj'
    
    # Call the calc_gso function with the provided adjacency matrix and gso_type
    computed_gso = calc_gso(adj, gso_type)
    
    computed_gso = computed_gso.toarray()
    computed_gso = computed_gso.astype(dtype=np.float32)
    computed_gso = torch.from_numpy(computed_gso)
    
    return computed_gso

def get_computed_gso():
    return torch.tensor([[0.5000, 0.3162, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.3162, 0.2000, 0.2582, 0.2582, 0.0000, 0.0000, 0.2582, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.2582, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.4082, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.2582, 0.0000, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.4082, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.4082, 0.5000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.4082,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4082, 0.5000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.4082, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.5000, 0.3162, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.3162, 0.2000, 0.2582, 0.2582, 0.0000, 0.0000, 0.2582, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2582, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.4082],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2582, 0.0000, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.4082, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4082, 0.5000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333,
         0.4082, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4082,
         0.5000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.4082, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.5000]])

def get_vae_stgcn(args=None, device='cpu'):
    if args is None:
        args = ModelConfig()
    
    # Initialize vae + stgcn
    encoder = Encoder(args, device)
    decoder = Decoder()
    vae_stgcn = VAE(encoder, decoder).to(device)
    
    return vae_stgcn

# reconstruction loss
def gaussian_likelihood(x_mean, x_var, x, device='cpu'):
    std = torch.exp(x_var / 2)
    dist = torch.distributions.Normal(x_mean, std)
    
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))
    
def kl_divergence(z, mu, log_var):
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)
    
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    
    return kl
  
def vae_loss(x, x_mean, x_var, z, mean, log_var, device):
    recon = gaussian_likelihood(x_mean, x_var, x, device=device)
    kl = kl_divergence(z, mean, log_var)
    
    return torch.mean(kl - recon)

# define predictor
class Predictor(nn.Module):
  def __init__(self, stgcn_blocks):
    super().__init__()
    self.stgcn_blocks = stgcn_blocks
    self.predict_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=65*20*32,
                        out_features=10400), # 9
        torch.nn.BatchNorm1d(10400),
        torch.nn.ReLU(),
        # torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features=10400, # 32 * 22 * 6
                        out_features=5200),
        torch.nn.BatchNorm1d(5200),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=5200,
                        out_features=9),
    )

  def forward(self, x):
    # x = x.permute(0,2,1,3).to(torch.float32)
    # x = x.permute(0,3,1,2).to(torch.float32)
    x = x.permute(0,2,1,3).to(torch.float32)
    output = self.predict_layer(self.stgcn_blocks(x))           
    output_probs = F.softmax(output, dim=1)
    
    return output_probs
    
def get_predictor(vae_stgcn, device='cpu'):
    stgcn_blocks = vae_stgcn.encoder.encoder[0]
    predictor = Predictor(stgcn_blocks).to(device)
    
    # freeze stgcn blocks parameters
    for param in predictor.stgcn_blocks.parameters():
        param.requires_grad = False
    
    return predictor