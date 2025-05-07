import os

import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
import scipy.io as sio


def load_from_mat(dataset,train_ratio,seed,path):

    filepath =  f'{path}/{dataset}.mat'
    mat_data = sio.loadmat(filepath)

    x = torch.tensor(mat_data['x'], dtype=torch.float)
    edge_index = torch.tensor(mat_data['edge_index'], dtype=torch.long)
    y = torch.tensor(mat_data['y'].flatten(), dtype=torch.long)  # 确保 y 是一维的

    data = Data(x=x, edge_index=edge_index, y=y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    split_path=f'{path}/split/{dataset}/split{train_ratio}_{seed}.mat'
    if not os.path.exists(split_path):
        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[:int(data.num_nodes * train_ratio)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * train_ratio):int(data.num_nodes * train_ratio * 2)])
        data.test_id = np.sort(node_id[int(data.num_nodes * (1 - 2 * train_ratio)):])

        if not os.path.exists(f'{path}/split/{dataset}'):
            os.makedirs(f'{path}/split/{dataset}/')
        sio.savemat(split_path,
                    {'train_id': data.train_id, 'val_id': data.val_id, 'test_id': data.test_id})
    else:
        split = sio.loadmat(split_path)
        data.train_id = split['train_id']
        data.val_id = split['val_id']
        data.test_id =  split['test_id']

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    data.num_nodes = x.size(0)

    return data



def get_raw_text(dataset, path='../data_TAG/'):
    csv_path =path+f'{dataset}_text.csv'
    text = []
    df = pd.read_csv(csv_path, encoding='utf-8')
    for index, row in df.iterrows():
        text.append(row['text'])

    return  text


def get_raw_text_instagram(dataset, path='../data_TAG/'):
    text = np.load(f'{path}/{dataset}_text.npy')
    return text.tolist()



def get_raw_TA(dataset,path):
    csv_path =f'{path}/{dataset}_TA_text.csv'
    text=[]
    df = pd.read_csv(csv_path, encoding='unicode_escape')
    for index, row in df.iterrows():
        ti = row['Title']
        ab = row['Abstract']
        text.append(str(ti) + '\n' + str(ab))

    return text