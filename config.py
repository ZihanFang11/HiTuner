import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):
    cfg.use_prompt = True
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    cfg.dataset = 'cora'
    cfg.data_path = "./data_TAG"

    cfg.device = "cuda:1"
    cfg.seed = None
    cfg.runs = 5
    cfg.gnn = CN()
    cfg.lm = CN()
    cfg.llm = CN()
    cfg.fusion = 'conf' #conf or average
    cfg.lr = 0.0001
    cfg.early_stop = 50
    cfg.weight_decay = 0.0
    cfg.dropout = 0.5

    cfg.thre = 0.1
    cfg.train_ratio = 0.1
    # Maximal number of epochs
    cfg.epochs = 300
    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name GCN,MLP,SAGE
    cfg.gnn.model.name = 'SAGE'
    # Number of gnn layers
    cfg.gnn.model.num_layers = 3
    # Hidden size of the model
    cfg.gnn.model.hidden_dim = 256

    # ------------------------------------------------------------------------ #
    # LLM Training options
    # ------------------------------------------------------------------------ #
    cfg.llm.name = "llama"
    cfg.llm.cache_path = "./llm_cache"
    cfg.llm.path = "./pretrain_models"
    cfg.llm.layer = 32
    cfg.llm.train = CN()
    cfg.llm.train.batch_size = 8
    cfg.llm.train.max_length = 512

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    cfg.lm.name = 'bert-base-uncased'
    cfg.lm.path = './pretrain_models/'
    cfg.lm.train.batch_size = 9
    cfg.lm.train.feat_shrink = ""
    cfg.lm.train.dim = 768
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.epochs = 4
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 50000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4

    return cfg




def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
