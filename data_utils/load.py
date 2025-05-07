
from load_data import load_from_mat
import torch


def load_data(dataset,train_ratio,seed, path):


    data = load_from_mat(dataset,train_ratio,seed, path)
    text = load_text(dataset,path)
    num_classes=torch.unique(data.y).size()[0]
    return data, num_classes, text

def load_text(dataset,path):
    if dataset == 'cora' or dataset == 'pubmed':
        from load_data import get_raw_TA as get_raw_text
    elif dataset == 'citeseer' or dataset == 'wikics' or  'photo' :
        from load_data import get_raw_text as get_raw_text
    elif dataset == 'instagram':
        from load_data import  get_raw_text_instagram  as get_raw_text
    else:
        exit(f'Error: Dataset {dataset} not supported')

    text = get_raw_text(dataset,path=path)
    return text