
import os
from data_utils.load import load_data
from time import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
import torch
from LMs.lm_trainer import LMTrainer
from config import cfg, update_cfg
import pandas as pd
def run(cfg):

    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    save_path = f"result_{cfg.dataset}.txt"

    all_acc = []
    all_time = []

    for seed in seeds:
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg.seed = seed
        data, num_classes, text = load_data(
            dataset=cfg.dataset, train_ratio=cfg.train_ratio, seed=cfg.seed, path= cfg.data_path)

        t0 = time()
        trainer = LMTrainer(cfg, data, num_classes, text)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)
        all_time.append(time()-t0)
        with open(save_path, "a") as f:
            for k in acc:
                f.write(f'{k}:{acc[k]*100:.2f} \t')
            f.write(f'{all_time[-1]:.4f}\n')
    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        with open(save_path, "a") as f:
            for k, v in df.items():
                print(f"{k}: {v.mean() * 100:.2f} ± {v.std() * 100:.2f} ")
                f.write(f"{k}: {v.mean() * 100:.2f} ± {v.std() * 100:.2f} ")
if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
