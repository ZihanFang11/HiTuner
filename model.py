import torch
from time import time
import numpy as np
import os
from data_utils.utils import EarlyStopping
from data_utils.utils import time_logger
import torch.nn as nn
LOG_FREQ = 10
from data_utils.utils import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.nn import SAGEConv



def get_hidden_states(config, select_layers):
    if config.use_prompt:
        path = f'{config.llm.cache_path}/{config.dataset}/layers/{config.llm.name}_prompt'
    else:
        path = f'{config.llm.cache_path}/{config.dataset}/layers/{config.llm.name}'

    if not os.path.exists(os.path.join(path, 'layer_attr.pt')):
        raise FileNotFoundError(
            f'No cache found! Please use `python cache.py --dataset {config.dataset}` to generate it.')
    else:
        layers_hid = torch.load(os.path.join(path, 'layer_attr.pt'))

    xs = [layers_hid[layer] for layer in select_layers]
    return xs




class HiModel(torch.nn.Module):
    def __init__(self,gnn_model_name, lm_dim, llm_dim, hidden_channels, out_channels, num_layers, thre, lamb,llm_layer,dropout):
        super(HiModel, self).__init__()
        out_channels=out_channels
        self.gnn_model_name = gnn_model_name
        if gnn_model_name=='GCN':
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(lm_dim, hidden_channels, cached=True))
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, cached=True))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        elif  gnn_model_name == "SAGE":
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(lm_dim, hidden_channels))
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        elif gnn_model_name == "MLP":
            self.convs = torch.nn.ModuleList()
            self.convs.append(nn.Linear(lm_dim, hidden_channels))
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.S_norm =  nn.BatchNorm1d(llm_dim, momentum=0.6)
        self.S =  torch.nn.ModuleList([  nn.Linear(llm_dim, lm_dim)for i in range(llm_layer)])
        self.theta = nn.Parameter(torch.FloatTensor([thre]), requires_grad=True)
        self.lamb = lamb
        self.ConfidNet=torch.nn.ModuleList([ nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  for i in range(llm_layer)])
        self.lm_emb_init = nn.Linear(llm_dim, lm_dim)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self,lm_emb, fea, adj_t):

        final=[]
        final_pred=[]
        for k in range(len(fea)):
            input1 = self.S[k](self.S_norm(fea[k]))
            llm_emb = self.soft_threshold(input1)
            x = lm_emb*self.lamb+llm_emb*(1-self.lamb)
            for i, conv in enumerate(self.convs[:-1]):
                if self.gnn_model_name == "MLP":
                    x = conv(x)
                else:
                    x = conv(x, adj_t)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            final.append(x)
            if self.gnn_model_name == "MLP":
                gnn_pred = self.convs[-1](x)
            else:
                gnn_pred = self.convs[-1](x, adj_t)

            final_pred.append(gnn_pred)
        return final, final_pred

    def soft_threshold(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)


class GNNTrainer():

    def __init__(self, cfg, data, num_classes, select_layers):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.dropout
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.fusion = cfg.fusion
        # ! Load data
        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        self.data = data.cuda()

        self.ckpt_dir = f'prt_lm_trainratio{cfg.train_ratio}/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}'

        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"

        # ! Trainer init
        xs = get_hidden_states(cfg, select_layers)
        self.select_layers = select_layers

        llm_hidden_emb = [x.to(torch.float32).cuda() for x in xs]
        self.features = llm_hidden_emb
        llm_dim = llm_hidden_emb[0].size(1)
        self.model = HiModel(self.gnn_model_name, lm_dim=cfg.lm.train.dim , llm_dim=llm_dim,
                             hidden_channels=self.hidden_dim,
                             out_channels=self.num_classes,
                             num_layers=self.num_layers, thre=cfg.thre, lamb=cfg.lamb, llm_layer=len(select_layers),
                             dropout=self.dropout,
                             ).cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=cfg.weight_decay)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")

        self.stopper = EarlyStopping(
            patience=cfg.early_stop, path=self.ckpt) if cfg.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

        self.feat_shrink =cfg.lm.train.feat_shrink

        print("Loading Fine-tuned PLM Embeddings as features...")
        emb = np.memmap(f"{self.ckpt_dir}.emb",
                        dtype=np.float16,
                        mode='r',
                        shape=(self.num_nodes, cfg.lm.train.dim))
        self.lm_emb = torch.from_numpy(emb).to(torch.float32).cuda()

    def FusionLayer(self, *predictions):

        predictions = predictions[0]
        if self.fusion == 'conf':
            mean_preds = [self.model.ConfidNet[i](predictions[i]) for i in range(len(predictions))]
            holo_values = []
            sum = mean_preds[0]
            for i in range(1, len(predictions)):
                sum = sum * mean_preds[i]
            for i in range(len(predictions)):
                tmp = sum / mean_preds[i]
                holo_values.append(torch.log(tmp) / (
                        torch.log(sum) + 1e-8))
            cb_factors = [mean_preds[i] * holo_values[i] for i in range(len(predictions))]
            w_all = torch.stack(cb_factors, 1)
            softmax = nn.Softmax(1)
            w_all = softmax(w_all)

            fusion_out = torch.stack([w_all[:, i] * predictions[i] for i in range(len(predictions))], dim=0).sum(dim=0)
        elif self.fusion == 'average':
            fusion_out = torch.stack([predictions[i] for i in range(len(predictions))], dim=0).mean(dim=0)
        else:
            exit(f'Error: Fusion {self.fusion} not supported')
        return fusion_out

    def _train(self):
        self.model.train()
        # criterion = nn.MSELoss()
        self.optimizer.zero_grad()
        gnn_embs, gnn_preds = self.model(self.lm_emb, self.features, self.data.edge_index)
        fusion_out = self.FusionLayer(gnn_preds)

        ce_loss = self.loss_func(
            fusion_out[self.data.train_mask], self.data.y[self.data.train_mask])
        print(f"ce_loss:{ce_loss:.8f}")
        loss = ce_loss
        train_acc = self.evaluator(
            fusion_out[self.data.train_mask], self.data.y[self.data.train_mask])

        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        gnn_embs, gnn_preds = self.model(self.lm_emb, self.features, self.data.edge_index)
        fusion_out = self.FusionLayer(gnn_preds)
        val_acc = self.evaluator(
            fusion_out[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            fusion_out[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, fusion_out

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time() - t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()

        print(
            f'[{self.gnn_model_name}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f},\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return res
