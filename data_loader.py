import random

import torch
from torch.utils import data
from utils import path_utils, train_utils


class Dataset(data.Dataset):
    def __init__(self, name_list, walk_id=0, need_sample=False):
        self.name_list = name_list
        self.need_sample = need_sample
        if walk_id is None:
            self.walk_id = ''
        else:
            self.walk_id = '_' + str(walk_id)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        case_name = self.name_list[index]
        node_feature_list = torch.load('{}/{}{}'.format(path_utils.node_data_path, case_name, self.walk_id))
        dense_raw_adj_list = torch.load('{}/{}'.format(path_utils.adj_data_path, case_name))
        edge_label = torch.load('{}/{}'.format(path_utils.label_data_path, case_name))
        return node_feature_list, dense_raw_adj_list, edge_label, case_name


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.tensor(mx.sum(dim=1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv).to_sparse()
    mx = torch.spmm(r_mat_inv, mx.squeeze()).unsqueeze(0)
    #
    # colsum = np.array(mx.sum(0))
    # c_inv = np.power(colsum, -1).flatten()
    # c_inv[np.isinf(c_inv)] = 0.
    # c_mat_inv = sp.diags(c_inv)
    # mx = mx.dot(c_mat_inv.dot(mx.T))
    return mx


def process_loaded_data(data, max_size, walk_id=None):
    node_feature_list, dense_raw_adj_list, edge_label, case_name = data
    N = edge_label.shape[1]

    if walk_id is None:
        node_feature_list = torch.Tensor(node_feature_list).view(1, N, -1)
    else:
        node_feature_list = torch.load('{}/{}_{}'.format(path_utils.node_data_path, case_name[0], str(walk_id)))
        node_feature_list = torch.Tensor(node_feature_list).view(1, N, -1)
    node_feature_list = node_feature_list.cuda()
    dense_normalized_adj_list = list(map(normalize, dense_raw_adj_list))
    # sparse_normalized_adj_list = list(map(lambda x: x.to_sparse(), dense_normalized_adj_list))
    dense_normalized_adj_list = train_utils.cuda_list_object_1d(dense_normalized_adj_list)
    edge_label = edge_label.view(1, N, N).cuda()
    if N > max_size:
        N = max_size
        node_feature_list = node_feature_list[:, :max_size]
        dense_raw_adj_list = [adj[:, :max_size, :max_size] for adj in dense_raw_adj_list]
        dense_normalized_adj_list = [adj[:, :max_size, :max_size] for adj in dense_normalized_adj_list]
        edge_label = edge_label[:, :max_size, :max_size]
    edge_label = edge_label.reshape(1, N * N)
    class_weight = torch.Tensor([1, float(N * N - edge_label.sum()) / float(edge_label.sum())]).cuda()
    return N, node_feature_list, dense_raw_adj_list, dense_normalized_adj_list, edge_label, class_weight, case_name[0]
