import multiprocessing
import pickle
from copy import copy

import numpy as np

from utils import path_utils, string_utils

# 为特征的数据进行标注：__name__feature 
def transform_feature(data, f_name, k=1):
    assert type(data) is list
    features = []
    for d in data:
        features.append("__%s__%s" % (f_name.upper(), d))
    return features

# 用于生成co_author 的特征信息
def gen_feature_author(name_list, reference_index):
    if reference_index >= 0:
        try:
            name_list.pop(reference_index)
        except Exception as e:
            print(len(name_list), reference_index)
            return []
    feature = []
    for name in name_list:
        feature.extend(transform_feature([string_utils.clean_name(name)], "name"))
    return feature

# 用于生成title 的特征信息
def gen_feature_title(title):
    if title:
        return transform_feature(string_utils.clean_sentence(title, stemming=True), 'title')
    else:
        return []

# 用于生成affiliation 的特征信息
def gen_feature_aff(aff):
    if aff:
        return transform_feature(string_utils.clean_sentence(aff), 'org')
    else:
        return []

# 用于生成venue 的特征信息
def gen_feature_venue(venue):
    if venue:
        return transform_feature(string_utils.clean_sentence(venue), 'venue')
    else:
        return []

# 用于生成keywords 的特征信息
def gen_feature_keywords(keywords):
    # print(keywords)
    # keywords = keywords.split(';')
    return transform_feature([string_utils.clean_name(k) for k in keywords], 'keyword')

# 生成paper 对，形成的格式为[(paper_id0, paper_id1), (paper_id0, paper_id2), (paper_id0, paper_id3), ...]
def extract_complete_pub_pair_list(pub_list):
    pub_pair_list = []
    for index_1, pub_1 in enumerate(pub_list):
        for index_2, pub_2 in enumerate(pub_list):
            pub_pair_list.append((pub_1, pub_2))
    return pub_pair_list

# 根据每个属性,为每个属性构建一个属性相似度graph
class FeatureSimilarityModel:
    def __init__(self, alph=0.5):
        self.idf_dict = pickle.load(open(path_utils.idf_dict_path, "rb"))
        self.idf_threshold = 10
        print("idf_threshold：{0}".format(str(self.idf_threshold)))

    def cal_pairwise_sim(self, raw_pub_list):
        pub_list = []
        for raw_pub in raw_pub_list:
            pub_list.append({
                'authors': gen_feature_author(raw_pub['authors'], raw_pub['reference_index']),
                'affiliation': gen_feature_aff(raw_pub['affiliation']),
                'title': gen_feature_title(raw_pub['title']),
                'venue': gen_feature_venue(raw_pub['venue']),
                'keywords': gen_feature_keywords(raw_pub['keywords']),
                # 'year': raw_pub['year'],
            })
        pub_pair_list = extract_complete_pub_pair_list(pub_list)
        pairwise_sim_list = []
        print('Computing pairwise similarity... ({}/{})'.format(0, len(pub_pair_list)))
        for index, pub_pair in enumerate(pub_pair_list):
            if (index + 1) % 100000 == 0:
                print('Computing pairwise similarity... ({}/{})'.format(index + 1, len(pub_pair_list)))
            pairwise_sim_list.append(self.cal_sim(*pub_pair))
        print('Computing pairwise similarity... ({}/{})'.format(len(pub_pair_list), len(pub_pair_list)))
        print('Applying adaptive masking...')
        return pairwise_sim_list
    
    def cal_sim(self, pub_1, pub_2):
        coauthor_sim = self.cal_idf_feature_sim(pub_1['authors'], pub_2['authors']) 
        affiliation_sim = self.cal_idf_feature_sim(pub_1['affiliation'], pub_2['affiliation'])
        title_sim = self.cal_idf_feature_sim(pub_1['title'], pub_2['title'])
        keywords_sim = self.cal_idf_feature_sim(pub_1['keywords'], pub_2['keywords'])
        venue_sim = self.cal_idf_feature_sim(pub_1['venue'], pub_2['venue'])
        # year_sim = self.cal_year_sim(pub_1['year'], pub_2['year'])
        all_feature_sim = self.cal_idf_feature_sim(pub_1['authors'] + pub_1['affiliation'] + pub_1['title'] + \
            pub_1['keywords'] + pub_1['venue'], pub_2['authors'] + pub_2['affiliation'] + pub_2['title'] + \
            pub_2['keywords'] + pub_2['venue']) #+ self.cal_year_sim(pub_1['year'], pub_2['year'])
        return [coauthor_sim, all_feature_sim]

    def cal_idf_feature_sim(self, feature_1, feature_2):
        if feature_2 is None or feature_1 is None or len(feature_1) == 0 or len(feature_2) == 0:
            return np.nan
        similarity = 0
        common_words = set(feature_1).intersection(set(feature_2))

        for word in common_words:
            similarity += self.idf_dict.get(word, self.idf_threshold) / (2 * self.idf_threshold)
        return similarity

    @staticmethod
    def cal_year_sim(year_1, year_2):
        if year_1 == 0 or year_2 == 0:
            return np.nan
        return 1 / (1 + abs(year_1 - year_2))