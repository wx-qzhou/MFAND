import json
import math
from utils import path_utils, data_utils
from collections import defaultdict as dd
from features import com_feature_similarity_model

# 超快
def read_raw_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as json_file:
        raw_data = json.load(json_file)
        return raw_data['pubs'], [raw_data['assignment'][group_id] for group_id in raw_data['assignment']]

pub = {}
pubs = {}
for file_index, file_name in enumerate(path_utils.train_raw_data_list + path_utils.test_raw_data_list):
        pub, _ = read_raw_data(file_name)
        pubs = dict(list(pub.items()) + list(pubs.items()))
        # pubs = pub

print(len(pubs))

features_list = []
authors_list = []
affiliation_list = []
title_list = []
venue_list = []
keywords_list = []

document_list = []
authors_list_all = []
affiliation_list_all = []
title_list_all = []
venue_list_all = []
keywords_list_all = []

for pub_id in pubs:
    if pubs[pub_id]['authors']:
        authors_list = com_feature_similarity_model.gen_feature_author(pubs[pub_id]['authors'], pubs[pub_id]['reference_index'])
        authors_list_all += authors_list
    if pubs[pub_id]['affiliation']:
        affiliation_list = com_feature_similarity_model.gen_feature_aff(pubs[pub_id]['affiliation'])
        affiliation_list_all += affiliation_list
    if pubs[pub_id]['title']:
        title_list = com_feature_similarity_model.gen_feature_title(pubs[pub_id]['title'])
        title_list_all += title_list
    if pubs[pub_id]['venue']:
        venue_list = com_feature_similarity_model.gen_feature_venue(pubs[pub_id]['venue'])
        venue_list_all += venue_list
    if pubs[pub_id]['keywords']:
        keywords_list = com_feature_similarity_model.gen_feature_keywords(pubs[pub_id]['keywords'])
        keywords_list_all += keywords_list

    features_list = authors_list + affiliation_list + title_list + venue_list + keywords_list
    document_list.append(features_list)

features_set_all = set(authors_list_all + affiliation_list_all + title_list_all + venue_list_all + keywords_list_all)
print(len(features_set_all))

counter = dd(int) 
for d_list in document_list:
    for f in set(d_list).intersection(features_set_all):
        counter[f] += 1  # 计算该特征的数量  counter = {feature: number}

print(len(counter))

cnt = len(document_list)
# 计算每个词的IDF
idf = dd(int)
for k in counter:
    idf[k] = math.log(cnt / (counter[k] + 1)) # idf = {feature： idf_value}

data_utils.dump_data(dict(idf), path_utils.feature_path, "feature_idf.pkl")