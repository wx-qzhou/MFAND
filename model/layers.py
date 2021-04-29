import math

import torch

from functools import reduce

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolutionWithEdgeConcat(Module):

    def __init__(self, in_features, out_features, edge_features, bias=True):
        super(GraphConvolutionWithEdgeConcat, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.share_weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = Parameter(torch.FloatTensor(in_features * edge_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.normal = torch.nn.LayerNorm(in_features, eps=1e-6)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.share_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_list):
        support_list = []
        for adj in adj_list:
            support_list.append(torch.spmm(adj, input))
        support = torch.cat(support_list, dim=1)
        support_ = reduce(torch.add, support_list)
        support_ = self.normal(support_)
        output = torch.mm(support, self.weight)
        output = (output + torch.mm(support_, self.share_weight)) / 2
        # output = torch.mm(support_, self.share_weight)
        del support_
        del input
        del adj_list
        del support
        del support_list
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionWithEdgeConcatparallel(Module):

    def __init__(self, in_features, out_features, n_edge_features, bias=False):
        super(GraphConvolutionWithEdgeConcatparallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = torch.nn.Linear(in_features * n_edge_features, out_features, bias=True)

    def forward(self, input, adj_list):
        # input, adj_list: (N， 16), (1，6, 1, N, N)
        support_list = []
        for adj in adj_list:
            support_list.append(torch.spmm(adj, input))
        support = torch.cat(support_list, dim=1)
        output = self.fc1(support)
        del support, support_list
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'