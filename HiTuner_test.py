import random
import os
import numpy as np
import torch
from time import time
import yaml
from yaml import SafeLoader
from config import cfg, update_cfg
import pandas as pd
from random import sample
import warnings
from data_utils.load import load_data
from model import GNNTrainer
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"




def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)


    cfg.config = f'./configs/{cfg.dataset}.yaml'
    args = yaml.load(open(cfg.config), Loader=SafeLoader)
    for k, v in args.items():
        cfg.__setattr__(k, v)
    print(cfg)

    save_path = f"result_{cfg.dataset}.txt"

    with open(save_path, "a") as f:
        f.write(f'\nllm:{cfg.llm.name},lm:{cfg.lm.name},lamb:{cfg.lamb} thre:{ cfg.thre}  lr:{cfg.lr} train_ratio:{cfg.train_ratio}\n')

    all_acc = []
    all_time = []
    for seed in seeds:

        t0 = time()
        cfg.seed = seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, num_classes, text = load_data(
            dataset=cfg.dataset,  train_ratio=cfg.train_ratio, seed=cfg.seed, path= cfg.data_path)

        layer = range(20,  cfg.llm.layer - 1)
        select_layers = sample(layer,  cfg.layer_num -1)
        select_layers = select_layers+[-1]

        trainer = GNNTrainer(cfg, data, num_classes,select_layers)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)
        all_time.append(time() - t0)
        with open(save_path, "a") as f:
            f.write(f'layer_num:{cfg.layer_num},select_layers:{select_layers}\n')
            for k in acc:
                f.write(f'{k}:{acc[k] * 100:.2f} \t')
            f.write(f'time:{all_time[-1]:.4f}\n')
        torch.cuda.empty_cache()
    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        with open(save_path, "a") as f:
            f.write(f'layer_num:{cfg.layer_num}\n')
            for k, v in df.items():
                print(f"{k}: {v.mean()*100:.2f} ± {v.std()*100:.2f} ")
                f.write(f"{k}: {v.mean()*100:.2f} ± {v.std()*100:.2f} ")

if __name__ == '__main__':
    cfg = update_cfg(cfg)

    run(cfg)

