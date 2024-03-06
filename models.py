import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import math

import numpy as np

# define an LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch=True):
        super(LSTMModel, self).__init__()
        self.batch = batch

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 150, num_layers, batch_first=True)
        if batch:
            self.batch_norm1 = nn.BatchNorm1d(hidden_size)
            self.batch_norm2 = nn.BatchNorm1d(150)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(150, 9)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1).float()
        # print(x.shape) # torch.Size([256, 130, 60])
        lstm_out, _ = self.lstm(x)
        if self.batch:
            lstm_out = self.batch_norm1(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)
        # print(lstm_out.shape) # torch.Size([256, 130, 250])
        lstm_out2, _ = self.lstm2(lstm_out)
        if self.batch:
            lstm_out2 = self.batch_norm2(lstm_out2.permute(0, 2, 1)).permute(0, 2, 1)
        # print(lstm_out2.shape) # torch.Size([256, 130, 150])
        lstm_out2 = lstm_out2[:, -1, :]
        # print(lstm_out2.shape) # torch.Size([256, 150])
        hidden_output = self.fc1(lstm_out2)
        # output_probs = F.softmax(hidden_output, dim=1)
        return hidden_output

def get_lstm(input_size=3*20, hidden_size=250, output_size=9, batch=True, device='cpu'):
    # Initialize the model
    lstm = LSTMModel(input_size, hidden_size, output_size, 1, batch=batch).to(device)

    return lstm

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
    self.graph = args.graph.to(device)

    modules = []
    
    modules.append(st_gcn(3, 64, (args.Kt, args.Ks), dropout=args.stgcn_dropout, graph=self.graph, residual=False))
    for _ in range(args.num_stgcn_blocks-1):
        modules.append(st_gcn(64, 64, (args.Kt, args.Ks), dropout=args.stgcn_dropout, graph=self.graph, residual=True))
    # modules.append(st_gcn(64, 64, (args.Kt, args.Ks), graph=self.graph, residual=True))
    # modules.append(st_gcn(64, 64, (args.Kt, args.Ks), graph=self.graph, residual=True))
    modules.append(nn.AvgPool2d(kernel_size=(2, 2)))
    st_blocks = nn.Sequential(*modules)

    self.encoder = torch.nn.Sequential(
        st_blocks,
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=65*20*32,
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
class STGCNConfig:
    def __init__(self, args=None):
        self.Kt = 9  # kernel for temporal convolution
        self.Ks = 3  # kernel for graph convolution
        self.num_stgcn_blocks = 4
        self.stgcn_dropout = 0
        self.graph = get_graph()

        if args is not None:
            self.set_configs(args=args)

    def set_configs(self, args):
        self.Kt = args["Kt"]
        self.Ks = args["Ks"]
        self.num_stgcn_blocks = args["num_stgcn_blocks"]
        self.stgcn_dropout = args["stgcn_dropout"]

def get_vae_stgcn(args=None, device='cpu'):
    if args is None:
        args = STGCNConfig()
    else:
        args = STGCNConfig(args=args)
    
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
  def __init__(self, stgcn_blocks, args):
    super().__init__()
    self.stgcn_blocks = stgcn_blocks
    if args["number_of_predictor_layers"] == 1:
        self.predict_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=65*20*32,
                            out_features=9),
        )
    else:
        self.predict_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=65*20*32,
                            out_features=10400),
            # torch.nn.BatchNorm1d(10400),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=10400,
                            out_features=5200),
            # torch.nn.BatchNorm1d(5200),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=5200,
                            out_features=9),
        )

  def forward(self, x):
    x = x.permute(0,2,1,3).to(torch.float32)
    output = self.predict_layer(self.stgcn_blocks(x))           
    # output_probs = F.softmax(output, dim=1)
    
    return output
    
def get_predictor(vae_stgcn, args=None, device='cpu'):
    stgcn_blocks = vae_stgcn.encoder.encoder[0]
    predictor = Predictor(stgcn_blocks, args).to(device)
    
    # freeze stgcn blocks parameters
    for param in predictor.stgcn_blocks.parameters():
        param.requires_grad = False
    
    return predictor