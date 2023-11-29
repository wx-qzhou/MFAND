import json
import os

import numpy as np
import torch
import random
import scipy.io
import networkx as nx
import scipy.sparse as sp
from features.com_feature_similarity_model import FeatureSimilarityModel
from utils import path_utils

print("mean / mean + max")
time = 5
print(time)
node_feature_size = 16 # 8, 12, 16, 20, 24, 28, 32, 36
feature_model = FeatureSimilarityModel()

def read_raw_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as json_file:
        raw_data = json.load(json_file)
        return raw_data['pubs'], [raw_data['assignment'][group_id] for group_id in raw_data['assignment']]

def extract_case_name(file_name):
    case_name = os.path.basename(file_name).split(".json")[0]
    return case_name

def extract_pub_id_list(pub_dict):
    pub_id_list = []
    for pub_id in pub_dict:
        pub_id_list.append(pub_id)
    return pub_id_list

def np_mx_to_sparse_mx(np_mx):
    np_mx = np_mx.reshape((np_mx.shape[0], np_mx.shape[1]))
    return sp.coo_matrix(np_mx,
                         dtype=np.float32)

def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse = torch.sparse.FloatTensor(indices, values, shape)
    dense = sparse.to_dense() 
    return dense

def extract_positive_pub_pair(assignment):
    positive_pair_list = []
    for group in assignment:
        for index_1, pub_id_1 in enumerate(group):
            for index_2, pub_id_2 in enumerate(group):
                if index_2 < index_1:
                    positive_pair_list.append((pub_id_1, pub_id_2))
    return positive_pair_list

def extract_negative_pub_pair(annotation_result):
    negative_pair_list = []
    for index_1, group_1 in enumerate(annotation_result): # [[], [], [], [], []]
        for index_2, group_2 in enumerate(annotation_result):
            if index_2 < index_1:
                for pub_id_1 in group_1:
                    for pub_id_2 in group_2:
                        negative_pair_list.append((pub_id_1, pub_id_2))
    return negative_pair_list

def save_gnn_data(
        adj_list,
        edge_label,
        allfeature,
        file_name,
):
    torch.save(
        adj_list,
        '{}/{}'.format(path_utils.adj_data_path, file_name)
    )
    torch.save(
        edge_label,
        '{}/{}'.format(path_utils.label_data_path, file_name)
    )
    torch.save(
        allfeature,
        '{}/{}'.format(path_utils.allfeature_data_path, file_name)
    )

def store_graph(file, edge_list):
    with open(file, "w") as f:
        for edge in edge_list:
            f.write(str(edge[0]))
            f.write(' ')
            f.write(str(edge[1]))
            f.write('\n')

def preprocess(pub_dict, assignment, case_name):  
    pub_id_list = extract_pub_id_list(pub_dict) # 获取paper_id
    raw_pub_list = [pub_dict[pub_id] for pub_id in pub_id_list] # 获取paper的具体内容 
    N = len(pub_id_list)
    print('Document num: {}'.format(N))
    edge_feature_list = np.array(feature_model.cal_pairwise_sim(raw_pub_list)) 
    n_edge_features = edge_feature_list.shape[-1] # 6
    edge_feature_list = edge_feature_list.reshape([N, N, n_edge_features])
    edge_feature_list = np.split(edge_feature_list, n_edge_features, axis=2)
    edge_feature_list = list(map(lambda x: x.squeeze(), edge_feature_list))

    edge_feature_list_ = []
    all_feature = []
    for edge_feature_index, feature_matrix in enumerate(edge_feature_list):
        feature_mean = np.nanmean(feature_matrix, axis=0) # 忽略这里面的nan，进行求均值1
        feature_matrix[np.isnan(feature_matrix)] = 0 # 将nan 赋值为0
        if edge_feature_index > 4:
            feature_mean = (time * feature_mean + np.max(feature_matrix, axis=0)) / (time + 1)
        # else:
            # feature_mean = (feature_mean + np.max(feature_matrix, axis=0)) / 2
        feature_matrix[feature_matrix < feature_mean + 1e-4] = 0 # 进行剪枝操作
        diag = np.copy(np.diag(feature_matrix)) # 形成对角矩阵
        diag_ = np.copy(diag)
        diag_[diag == 0] = 1
        feature_matrix -= np.diag(diag) # 去掉对角元素
        feature_matrix += np.diag(diag_) # 用1 进行填充
        feature_matrix += np.transpose(feature_matrix) # 矩阵进行转置
        feature_matrix /= 2 # 对原矩阵进行归一话操作
        if edge_feature_index <= 4:
            edge_feature_list_.append(feature_matrix)
        else:
            all_feature = feature_matrix
            edge_list = np.transpose(np.nonzero(feature_matrix)).tolist()
            store_graph(path_utils.edge_data_path + '/' + case_name + '.txt', edge_list)
    edge_feature_list = edge_feature_list_

    adj_list = list(map(np_mx_to_sparse_mx, edge_feature_list))
    adj_list = list(map(sparse_mx_to_torch_tensor, adj_list))

    positive_pub_id_pair_list = extract_positive_pub_pair(assignment)
    negative_pub_id_pair_list = extract_negative_pub_pair(assignment)

    print('Generating edge labels...')
    label_dict = {"{}_{}".format(pub_id_1, pub_id_2): 1 for pub_id_1, pub_id_2 in positive_pub_id_pair_list} # 如果两个节点之间存在边那么这两个节点之间形成的边为1，否则为0
    label_dict.update(
        {"{}_{}".format(pub_id_2, pub_id_1): 1 for pub_id_1, pub_id_2 in positive_pub_id_pair_list})
    label_dict.update(
        {"{}_{}".format(pub_id_1, pub_id_2): 0 for pub_id_1, pub_id_2 in negative_pub_id_pair_list})
    label_dict.update(
        {"{}_{}".format(pub_id_2, pub_id_1): 0 for pub_id_1, pub_id_2 in negative_pub_id_pair_list})
    edge_label = torch.zeros_like(adj_list[0]).long() # 其维度与矩阵adj_list[0]一致，并为其初始化为全0
    for col_index in range(0, N):
        for row_index in range(0, N):
            if col_index != row_index:
                pub_id_1 = pub_id_list[col_index]
                pub_id_2 = pub_id_list[row_index]
                edge_label[col_index][row_index] = label_dict["{}_{}".format(pub_id_1, pub_id_2)]
            else:
                edge_label[col_index][row_index] = 1
    
    return adj_list, edge_label, all_feature

def preprocess_all():
    for file_index, file_name in enumerate(path_utils.train_raw_data_list + path_utils.test_raw_data_list):
        case_name = extract_case_name(file_name)
        if case_name not in path_utils.preprocessed_data_list: # 生成每个author的的处理后的数据
            print('Reading {}... ({}/{})'.format(
                case_name,
                file_index + 1,
                len(path_utils.train_raw_data_list + path_utils.test_raw_data_list)
            ))
            processed_data = preprocess(*read_raw_data(file_name), case_name)
            save_gnn_data(
                *processed_data,
                case_name,
            )


if __name__ == '__main__':
    preprocess_all()