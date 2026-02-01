#  HiTuner: Hierarchical Semantic Fusion Model Fine-Tuning on Text-Attributed Graphs

This repository is an implementation of HiTuner in IJCAI 2025.

## Configure the Setup:

The configuration parameters are stored in the `config.py` file. You can customize them as needed for your specific setup.

    ```
    # Setup configuration
    cfg.dataset = 'cora'
    cfg.llm.name = "llama"
    cfg.lm.name = 'bert-base-uncased'
    cfg.gnn.model.name = 'SAGE'

    ```
   

## Datasets:

The datasets used can be downloaded from [here](https://drive.google.com/drive/folders/1MUx97je9je2MMDJGWxtc1S4Pl8DEdqS0), 
please download them and put them in datasets to `cfg.data_path`.

| Dataset    | # Nodes | # Edges | # Clusters | Domain        |
|------------|---------|---------|------------|---------------|
| CiteSeer   | 3,186   | 4,277   | 6          | Academic      |
| Cora       | 2,708   | 5,429   | 7          | Academic      |
| Instagram  | 11,339  | 144,010 | 2          | Social        |
| Photo      | 48,362  | 500,928 | 12         | E-commerce    |
| PubMed    | 19,717  | 44,338  | 3          | Academic      |
| WikiCS    | 11,701  | 216,123 | 10         | Wikipedia     |



## Runing Commands

### Step 1. 
```
python3 LLM_infer.py 
```
The cache.py will load the textual data of TAG, and next transform them to token embedding by LLM, which will be saved into `cfg.llm.cache_path`. 


### Step 2. 
```
python3 LM_train.py 
```
LM_train.py  will divide the dataset into nodes according to the training rate and save the divided nodes. 
Then it will be updated through small-scale pre-trained model training and saved in output_trainratio{`cfg.train_ratio`}.
The saved embeddings will used in the training of HiTuner.


### Step 3. 

```
python3 HiTuner_test.py 
```
After preprocessing the dataset, we run HiTuner for downstream tasks.



## Citation
If you find our work or dataset useful, please consider citing our work:
```
@inproceedings{hituner,
  title={HiTuner: Hierarchical Semantic Fusion Model Fine-Tuning on Text-Attributed Graphs},
  author={Zihan Fang and Zhiling Cai and Yuxuan Zheng and Shide Du and Yanchao Tan and Shiping Wang},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  year={2025}
}
```
