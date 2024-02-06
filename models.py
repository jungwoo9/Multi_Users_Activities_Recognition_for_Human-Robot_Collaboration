import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
class Encoder(torch.nn.Module):
  def __init__(self, args, device):
    super().__init__()

    self.device = device

    modules = []
    modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=3, channels=[64, 64, 64], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=64, channels=[64, 64, 64], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=64, channels=[64, 64, 64], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    modules.append(STConvBlock(Kt=args.Kt, Ks=args.Ks, n_vertex=args.n_vertex, last_block_channel=64, channels=[64, 64, 32], act_func=args.act_func, graph_conv_type=args.graph_conv_type, gso=args.gso, bias=args.enable_bias, droprate=args.droprate))
    modules.append(nn.AvgPool2d(kernel_size=(3, 3)))
    st_blocks = nn.Sequential(*modules)

    self.encoder = torch.nn.Sequential(
        st_blocks,
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=32 * 22 * 6,
                        out_features=2600),
        torch.nn.ReLU()
    )

    # mean and sigma layer
    self.z_mean = torch.nn.Linear(in_features=2600,
                                  out_features=1950)
    self.z_log_var = torch.nn.Linear(in_features=2600,
                                     out_features=1950)

  def reparameterization(self, z_mean, z_log_var):
    epsilon = torch.normal(mean=0, std=1.0, size=(z_mean.shape[0], z_mean.shape[1])).to(self.device)

    return z_mean + torch.exp(0.5 * z_log_var) * epsilon

  def forward(self, x):
    x = x.permute(0,2,1,3).to(torch.float32)
    x = self.encoder(x)
    z_mean = self.z_mean(x)
    z_log_var = self.z_log_var(x)
    z = self.reparameterization(z_mean, z_log_var)

    return z, z_mean, z_log_var
    
# define decoder
class Decoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.decoder_hidden = torch.nn.Sequential(
        torch.nn.Linear(in_features=1950,
                        out_features=3900),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=3900,
                        out_features=7800),
        torch.nn.Tanh()
    )

  def forward(self, x):
    x = self.decoder_hidden(x)
    x = x.reshape(x.shape[0], 130, 3, 20)
    return x

# define vae
class VAE(torch.nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder =decoder

  def forward(self, x):
    z, z_mean, z_log_var = self.encoder(x)
    return self.decoder(z), z_mean, z_log_var

# define configurations for stgcn
class ModelConfig:
    def __init__(self, computed_gso):
        self.Kt = 9  # kernel for temporal convolution
        self.Ks = 3  # kernel for graph convolution
        self.n_his = 130  # number of historical time steps
        self.act_func = 'glu' 
        self.n_vertex = 20 # number of vertex in data
        self.graph_conv_type = 'graph_conv'
        self.gso = computed_gso  # graph shift operator | adjacent matrix
        self.enable_bias = True  # enable bias terms in linear layers
        self.droprate = 0.3

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
        computed_gso = get_gso()
        args = ModelConfig(computed_gso.to(device))
    
    # Initialize vae + stgcn
    encoder = Encoder(args, device)
    decoder = Decoder()
    vae_stgcn = VAE(encoder, decoder).to(device)
    
    return vae_stgcn

# loss function for vae
def vae_loss(x, recon_x, mean, log_var):
  # 1. Reconstrut loss : Cross-entropy
   reconstruction_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#   reconstruction_loss = torch.nn.MSELoss()(recon_x, x)

  # 2. KL divergence(Latent_loss)
  KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

  return reconstruction_loss + KLD

# define predictor
class Predictor(nn.Module):
  def __init__(self, stgcn_blocks):
    super().__init__()
    self.stgcn_blocks = stgcn_blocks
    self.predict_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=32 * 22 * 6,
                        out_features=9)
    )

  def forward(self, x):
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