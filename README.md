# MetGen: A Module-based Entailment Tree Generation Framework for Answer Explanation

# Introduction
An implementation for `MetGen: A Module-based Entailment Tree Generation Framework for Answer Explanation`.
Please contact [Ruixin Hong](hrx20@mails.tsinghua.edu.cn) for questions and suggestions.


# Requirements
- Python3.8
- Ubuntu21.04
- Python Packages
```
conda create -n metgen python=3.8
conda activate metgen
pip install -r requirements.txt
```

# Data
Download [EntailmentBank dataset](https://allenai.org/data/entailmentbank).

Download [preprocessed data](this link).

`data/` folder contains four files：

```
data
├── entailment_trees_emnlp2021_data_v2  # the EntailmentBank dataset
├── wiki_match                          # the synthetic data for module training
├── Steps                               # the annotated/pseudo step data for module training
└── Controller_data                     # the processed data for controller training
```

# Training

## Single-step entialment module

Follow the `./scripts/train_module.sh` to train the prefixed module.
- Step1: train the module with synthetic data
- Step2: train the module with Train-pseudo data

The trained module would be saved in the `/exp/Module_all/para_etree_all` folder.


## Reasoning controller
Follow the `./scripts/train_controller.sh` to train the reasoning controller.

- Step1: make controller training data based on the orginal dataset and the trained module
- Step2: train the controller with the data

We train two controllers, one for Task 1 and one for Task 2/3.
The trained controller would be save in the `/exp/Controller_task1` and `/exp/Controller_task2` folders.

# Evaluation

## Inference
Follow the `./scripts/test_task1.sh`, `./scripts/test_task2.sh`, and `./scripts/test_task3.sh` to obtain the predictions based on the trained module and controllers.

- Step1: select the checkpoint and the hyperparameters of reasoning algorithm using the dev split 
- Step2: run the reasoning algorithm with the selected checkpoint and hyperparameters on the test split

The predictions would be save in `.json` and `.csv` files.

## Automatic Metric
Use the `.csv` file and follow the [offical evaluation code of EntailmentBank](https://github.com/allenai/entailment_bank) to evaluate automatically.

For Task 1 and Task 2, we also provide the evaluation results by our implementation of the evaluation code (`code/evaluate_metric.py`).

# Trained Models

We provide the [trained models](this link) for direct reproduction.
Unzip the file and place it in `exp` folder.
Run the following commands to reproduce the results.
```
cd scripts
sh test_task1.sh
sh test_task2.sh
sh test_task3.sh
```



# Citation
The citation is coming up soon.