import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import GraphConvolutionWithEdgeConcat, GraphConvolutionWithEdgeConcatparallel
from utils import train_utils

"""该处形成的是三元组对"""
class ConcatDecoder(nn.Module):
    """Decoder for using MLP for prediction."""

    def __init__(self, dropout, n_node_features, hidden_1, hidden_2, n_edge_features, n_classes=1):
        super(ConcatDecoder, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes

    def forward(self, z, raw_adj_list):
        N = raw_adj_list[0].shape[0]
        z = F.dropout(z, self.dropout, training=self.training)
        raw_adj_list = [adj.view(N, N, 1) for adj in raw_adj_list]
        raw_adj_list = torch.cat(raw_adj_list, dim=2)
        adj = torch.cat([z.repeat(1, N).view(N * N, -1), z.repeat(N, 1)], dim=1).view(N, N, -1)
        adj = torch.cat([adj, raw_adj_list], dim=2)
        del z, raw_adj_list
        return adj


class MLP(nn.Module):
    """Decoder for using MLP for prediction."""

    def __init__(self, dropout, n_node_features, hidden_1, hidden_2, n_edge_features, n_classes=1):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes
        n_input = n_node_features * 2 + n_edge_features
        self.fc1 = nn.Linear(n_input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, n_classes)

    def forward(self, adj):
        adj = F.relu(self.fc1(adj), inplace=True)
        adj = F.dropout(adj, self.dropout, training=self.training)
        frame = inspect.currentframe()
        adj = F.relu(self.fc2(adj), inplace=True)
        adj = F.dropout(adj, self.dropout, training=self.training)
        adj = self.fc3(adj)
        adj = F.log_softmax(adj, dim=2).view(-1, self.n_classes)
        del frame
        return adj


class GCN(nn.Module):
    def __init__(self, n_node_features, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                 n_classes, n_edge_features, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        # self.gcp = GraphConvolutionWithEdgeConcatparallel(n_node_features, n_hidden_1, n_edge_features)
        self.gc1 = GraphConvolutionWithEdgeConcat(n_node_features, n_hidden_1, n_edge_features)
        self.gc2 = GraphConvolutionWithEdgeConcat(n_hidden_1, n_hidden_2, n_edge_features)
        self.decoder = ConcatDecoder(dropout, n_hidden_2, n_hidden_3, n_hidden_4, n_edge_features, 
                                     n_classes=n_classes)
        self.mlp = MLP(dropout, n_hidden_2, n_hidden_3, n_hidden_4, n_edge_features, n_classes=n_classes)

    def encode(self, x, adj_list):
        y = F.relu(self.gc1(x, adj_list), inplace=True)
        # y = F.dropout(y, self.dropout, training=self.training)
        z = self.gc1(y, adj_list)
        # z = F.dropout(z, self.dropout, training=self.training)
        # x = F.relu((y + z) / 2, inplace=True)
        x = (y + z)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_list)
        del y, z
        return x

    def decode(self, *args):
        adj = self.decoder(*args)
        return adj

    def forward(self, x, adj_list, raw_adj_list):
        x = x.squeeze()
        adj_list = list(map(lambda x: x.squeeze(0), adj_list))
        raw_adj_list = list(map(lambda x: x.squeeze(0), raw_adj_list))
        embedding = self.encode(x, adj_list)
        raw_adj_list = train_utils.cuda_list_object_1d(raw_adj_list)
        del x
        del adj_list
        output = self.decode(embedding, raw_adj_list)
        del embedding
        del raw_adj_list
        output = self.mlp(output)
        return output
