import torch
import torch.nn as nn
from torch import optim

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

#logger.disabled = True  #暫停 logger
#logger.handlers  # logger 內的紀錄程序
#logger.removeHandler  # 移除紀錄程序
#logger.info('xxx', exc_info=True)  # 紀錄堆疊資訊
def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)

    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)

    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    return my_logger

def del_logger(logger):
    # logger.disabled = True  #暫停 logger
    while logger.handlers:  # logger 內的紀錄程序
        h = logger.handlers[0]
        logger.removeHandler(h)

# 資料整理:
# 一份資料一個月
# 每一個row 為 1 名 user , user feature, 前三名item的金額
# 暫定 user_feature: ['masts', 'educd', 'trdtp', 'poscd', 'gender_code', 'age']
#     user_performance: ['chid', 'shop_tag', 'txn_amt']
# txn: ['txn_cnt', 'txn_amt']

def data_transform(data, user_id, user_feature, items):
    l = [2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]  #消費類別對應的 index

    N = len(user_id)
    features = np.zeros((N, len(user_feature)))
    txn_amt = np.zeros((N, 16))  # 尚未寫
    labels = np.zeros((N, 6))
    for i in range(N):
        # i = 354952
        idx = user_id[i]
        d = data[data['chid'] == idx]

        # 得到 user features 結果
        if d.shape[0] != 0:
            feature = d.iloc[0:1, :][user_feature].to_numpy()
            features[i:i+1, :] = feature

        d = d[d['shop_tag'].isin(items)]
        d = d.sort_values(by=['txn_amt'], ascending=False)
        shop_tags = d['shop_tag'].to_list()
        shop_values = d['txn_amt'].to_list()

        # 得到top3 shop tag和他們的消費金額
        for j, c in enumerate(shop_tags[:3]):
            labels[i, j] = c
        for j, c in enumerate(shop_values[:3]):
            labels[i, j+3] = c

        # 得到每類別的消費金額
        for a, b in zip(shop_tags, shop_values):
            j = l.index(int(a))
            txn_amt[i, j] = b
        # return d, features, txn_amt, labels
        
    # 得到 user index list
    uid = user_id.to_numpy().reshape(-1, 1)
    outputs = [uid, txn_amt, features, labels]
    outputs = np.concatenate(outputs, axis=1)
    return outputs

'''
Ql features:
masts: [0., 1., 2., 3.]
educd: [0., 1., 2., 3., 4., 5., 6.]
trdtp:[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
       26., 27., 28., 29.]
poscd: [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 99.]
gender_code: [0., 1.]
age: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
labels: [0, 2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]
'''
# use to transform values from array with its Ql feature for inputs to model
def inputs_trans(datas):
    def tran(d):
        m = max(d)
        outputs = [-1] * (int(m) + 1)
        for i, v in enumerate(d):
            outputs[int(v)] = i
        return np.array(outputs)

    masts = [0., 1., 2., 3.]
    educd = [0., 1., 2., 3., 4., 5., 6.]
    trdtp = [0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
            13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
            26., 27., 28., 29.]
    poscd = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 99.]
    gender_code = [0., 1.]
    age = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    labels = [0, 2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]

    l = [masts, educd, trdtp, poscd, gender_code, age, labels, labels, labels]
    l = list(map(tran, l))

    _, _, b = datas.shape
    outputs = [datas[:, :, 0:17]]
    for i in range(17, b-3):
        c1 = datas[:, :, i:i+1].astype('int')
        c2 = l[i-17]
        outputs.append(c2[c1])
    outputs.append(datas[:, :, -3:])
    outputs = np.concatenate(outputs, axis=2)
    return outputs

