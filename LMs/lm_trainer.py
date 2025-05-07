
import torch
import numpy as np
from data_utils.utils import Evaluator
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy, BertTokenizer
from LMs.model import BertClassifier, BertClaInfModel
from data_utils.dataset import Dataset
from utils import init_path, time_logger
def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

from torch.utils.data import DataLoader
class LMTrainer():
    def __init__(self, cfg,data, num_classes, text):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.name
        self.feat_shrink = cfg.lm.train.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.output_dir = f'output_trainratio{cfg.train_ratio}/{self.dataset_name}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm_trainratio{cfg.train_ratio}/{self.dataset_name}/{self.model_name}-seed{self.seed}'

        # Preprocess data

        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes
        self.model_path = cfg.lm.path + self.model_name

        model = AutoModel.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        X = tokenizer(text, padding=True, truncation=True, max_length=512)

        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        self.train_loader = DataLoader( self.train_dataset , batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader( self.val_dataset, batch_size=self.batch_size * 8)

        self.model = BertClassifier(model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")


    @time_logger
    def train(self):

        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,  # Path to save model checkpoints
            do_train=True,  # Enable training
            do_eval=True,  # Enable evaluation during training
            eval_steps=eval_steps,  # Evaluate every `eval_steps` steps
            evaluation_strategy=IntervalStrategy.STEPS,  # Evaluation is triggered by step intervals
            save_steps=eval_steps,  # Save model checkpoint every `eval_steps` steps
            learning_rate=self.lr,  # Initial learning rate
            weight_decay=self.weight_decay,  # L2 regularization strength
            save_total_limit=1,  # Keep only the most recent checkpoint
            load_best_model_at_end=True,  # Automatically load the best checkpoint at the end of training
            gradient_accumulation_steps=self.grad_acc_steps,
            # Accumulate gradients over multiple steps to simulate a larger batch
            per_device_train_batch_size=self.batch_size,  # Batch size per GPU/CPU for training
            per_device_eval_batch_size=self.batch_size * 8,  # Batch size per GPU/CPU for evaluation
            warmup_steps=warmup_steps,  # Number of warmup steps for learning rate scheduling
            num_train_epochs=self.epochs,  # Total number of training epochs
            dataloader_num_workers=1,  # Number of worker threads for data loading
            fp16=True,  # Use mixed precision training (FP16) for faster computation
            dataloader_drop_last=True,  # Drop the last batch if it’s smaller than the specified batch size
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,  # Training arguments defined above
            train_dataset=self.train_dataset,  # Dataset used for training
            eval_dataset=self.val_dataset,  # Dataset used for evaluation
            compute_metrics=compute_metrics,  # Metric computation function during evaluation
        )
        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model,emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)



        _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))


        inf_model = BertClaInfModel(
            self.model,emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)



        _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}