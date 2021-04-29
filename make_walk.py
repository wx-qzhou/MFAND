import json
import os
import torch
import random
import numpy as np
import networkx as nx
import scipy.sparse as sp
from utils import path_utils

node_feature_size = 16 # 8, 12, 16, 20, 24, 28, 32, CA
walk_length = 20

def read_raw_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as json_file:
        raw_data = json.load(json_file)
        return raw_data['pubs'], [raw_data['assignment'][group_id] for group_id in raw_data['assignment']]

def extract_case_name(file_name):
    case_name = os.path.basename(file_name).split(".json")[0]
    return case_name

def save_gnn_data(
        node_walk,
        file_name,
        index,
):
    torch.save(
        node_walk,
        '{}/{}'.format(path_utils.node_data_path, file_name + '_' +str(index))
    )
    
class Create_Path(object): 
    def __init__(self, case_name): 
        self.G = nx.read_edgelist(os.path.join(path_utils.edge_data_path + '/' + case_name + '.txt'), create_using = nx.Graph()) # edges的文件的路径
        self.walk_len = node_feature_size
        self.return_prob = 0
        
    def generate_random_walks(self, rand=random.Random(0)):
        """generate random walks
        """
        walks = list()
        node_id = 0
        for node in list(self.G.nodes()):
            walk_ = self.random_walk(rand=rand, start=node, node_id=node_id)
            walk_add_len = self.walk_len - len(walk_)
            if walk_add_len != 0:
                walk_ += [node_id] * walk_add_len
            walks.append(walk_)
            node_id += 1
        return walks
        
    def random_walk(self, node_id, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        if start is None:
            print("random walk need a start node!")
        path = [start]

        cur_path_length = self.walk_len
        while len(path) < cur_path_length:
            cur = path[-1]
            if len(list(self.G.neighbors(cur))) > 0:
                if rand.random() >= self.return_prob:
                    path.append(rand.choice(list(self.G.neighbors(cur))))
                else:
                    path.append(path[0])
            else:
                break
        return [[node_id, int(node)] for node in path]


def preprocess(create_Path, allfeature, pub_len, case_name):   
    node_feature_list = [[]] * pub_len
    walk = create_Path.generate_random_walks()
    walk = np.array(walk).reshape(-1, 2)
    index = list(zip(*walk))
    node_feature_list = allfeature.squeeze()[tuple(index)]
    node_feature_list = node_feature_list.reshape(1, pub_len, -1).tolist()
    # for pub_id in range(0, pub_len):
        # node_feature = []
        # for node_id in walk[pub_id]:
            # node_feature.append(allfeature[pub_id][int(node_id)])
        # node_feature_list[pub_id] = np.array(node_feature)
    return node_feature_list
    
def preprocess_all():
    for file_index, file_name in enumerate(path_utils.train_raw_data_list + path_utils.test_raw_data_list):
        case_name = extract_case_name(file_name)
        if case_name in path_utils.preprocessed_data_list: # 生成每个author的的处理后的数据
            print('Reading {}... ({}/{})'.format(
                case_name,
                file_index + 1,
                len(path_utils.train_raw_data_list + path_utils.test_raw_data_list)
            ))
            pub_dict, _ = read_raw_data(file_name)
            pub_len = len(pub_dict)
            create_Path = Create_Path(case_name)
            allfeature = torch.load('{}/{}'.format(path_utils.allfeature_data_path, case_name))
            for index in range(walk_length):
                processed_data = preprocess(create_Path, allfeature, pub_len, case_name)
                save_gnn_data(
                    *processed_data,
                    case_name,
                    index,
                )
if __name__ == '__main__':
    preprocess_all()