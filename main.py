import torch
import torch.nn as nn
from torch import optim

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils

# data split
def data_split():
    path = './dataset/tbrain_cc_training_48tags_hash_final.csv'
    data_chunks = pd.read_csv(path)

    for i in range(1, 25):
        spath = f'./dataset/date_data/dt{i}.csv'
        c = data_chunks[data_chunks['dt'] == i]
        c.to_csv(spath)

# data features:
# [id, tag1_txn~tag16_txn, masts, educd, trdtp, poscd, gender_code, age, top1_tag~top3_tag, top1_tag_txn~top3_tag_txn]
def data_transform():
    path = './dataset/date_data/'
    idpath = './dataset/data_information/id_list.csv'
    ids = pd.read_csv(idpath)['chid']

    features = ['chid', 'shop_tag', 'txn_amt', 'dt', 'masts', 'educd', 'trdtp', 'poscd', 'gender_code', 'age']
    shop_tag = ['2', '6', '10', '12', '13', '15', '18', '19', '21', '22', '25', '26', '36', '37', '39', '48']

    user_feature = ['masts', 'educd', 'trdtp', 'poscd', 'gender_code', 'age']
    user_performance = ['chid', 'shop_tag', 'txn_amt']

    try:
        for i in range(1, 25):
            dat = f'dt{str(i)}.csv'
            p = path + dat
            data = pd.read_csv(p)[features]
            outputs = utils.data_transform(data, ids, user_feature, shop_tag)
            with open(f'./dataset/user_features3/dt{str(i)}.pickle', 'wb') as f:
                pickle.dump(outputs, f)
    except Exception as e:
        print(str(e))
        with open(f'./dataset/user_features3/wrong{str(i)}', 'wb') as f:
                pickle.dump([0], f)

# 得到的資料再作一次較小的變換，把當月沒有消費的user特徵用上一個月份的補
def data_trans():
    path = './dataset/user_features3'
    datas = []
    for i in range(1, 25):
        datas.append(path + f'/dt{str(i)}.pickle')

    for i, p in enumerate(datas):
        with open(p, 'rb') as f:
            a = pickle.load(f)
            a = np.nan_to_num(a)
            datas[i] = a

    n = len(datas)
    for i in range(1, n):
        a1, a2 = datas[i-1][:, 17:23], datas[i][:, 17:23]
        t = (a2 == 0) + 0.
        datas[i][:, 17:23] = a2 + a1 * t

    for i in range(n-2, -1, -1):
        a1, a2 = datas[i+1][:, 17:23], datas[i][:, 17:23]
        t = (a2 == 0) + 0.
        datas[i][:, 17:23] = a2 + a1 * t

    for i, d in enumerate(datas):
        with open(f'./dataset/user_features4/dt{i+1}.pickle', 'wb') as f:
            pickle.dump(d, f)

# user id with feature extract
# def id_feature():
#     featureList = ['chid', 'masts', 'educd', 'trdtp', 'poscd', 'gender_code', 'age']
#     path = './dataset/tbrain_cc_training_48tags_hash_final.csv'
#     d = pd.read_csv(path)
#     d = d[featureList]

#     idpath = './dataset/data_information/id_list.csv'
#     ids = pd.read_csv(idpath)['chid']

#     outputs = []
#     for i in range(len(ids)):
#         idx = ids[i]
#         c = d[d['chid'] == idx].tail(1)
#         outputs.append(c)

#     chids = []
#     outputs = []
#     for i in tqdm(range(len(d.index), 0, -1)):
#         ind = i - 1
#         chid = int(d.loc[ind]['chid'])
#         if chid not in chids:
#             outputs.append(d.loc[ind:ind])
#             chids.append(chid)
#     outputs = pd.concat(outputs, ignore_index=True)
#     outpus.to_csv('./dataset/user_features/id_features.csv')