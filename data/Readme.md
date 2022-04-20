## Introduction

The data for `MetGen: A Module-based Entailment Tree Generation Framework for Answer Explanation`.


```
data
├── Controller_data                         # the processed data for controller training
│   ├── dev.controller.task1.v36.jsonl
│   ├── dev.controller.task2.v36.jsonl
│   ├── train.controller.task1.v36.jsonl    # the training data for Task1
│   └── train.controller.task2.v36.jsonl
├── entailment_trees_emnlp2021_data_v2      # the EntailmentBank dataset
│   ├── dataset
│   │   ├── task_1
│   │   │   ├── dev.jsonl
│   │   │   ├── test.jsonl
│   │   │   └── train.jsonl
│   │   ├── task_2
│   │   │   ├── dev.jsonl
│   │   │   ├── test.jsonl
│   │   │   └── train.jsonl
│   │   └── task_3
│   │       ├── dev.jsonl
│   │       ├── test.jsonl
│   │       └── train.jsonl
│   └── supporting_data
│       └── worldtree_corpus_sentences_extended.json
├── Steps                               # the annotated/pseudo step data for module training
│   ├── Dev_manual                      # the annotated steps from the EntailmentBank dev split
│   │   └── inference_type_labeled_dev_data.jsonl
│   ├── Train_manual                    # the annotated steps from the EntailmentBank training split
│   │   └── inference_type_labeled_train_data_400.jsonl
│   └── Train_pseudo                    # the steps with pseudo labels from the EntailmentBank training split
│       ├── pseudo.all.jsonl
│       ├── pseudo.conjunction.jsonl
│       ├── pseudo.if-then.jsonl
│       └── pseudo.substitution.jsonl
└── wiki_match                          # the synthetic data
    └── V1
        ├── Conjunction                 # the Conjunction steps
        │   ├── dev.jsonl               # from wikipidia
        │   ├── test.jsonl              # test data is from Dev_manual
        │   └── train.jsonl             # from wikipidia
        ├── Ifthen
        │   ├── dev.jsonl
        │   ├── test.jsonl
        │   └── train.jsonl
        └── Substitution
            ├── dev.jsonl
            ├── test.jsonl
            └── train.jsonl
```

## Data for module training
- `wiki_match` is the synthetic data collected from Wikipidia following [ParaPattern](https://arxiv.org/abs/2104.08825).
It contains three types of entailmen steps: Substitution, Conjunction, and Ifthen.
For each type, the `train.jsonl` and `dev.jsonl` are synthetic data, while the `test.jsonl` is the data from the `Dev-manual`.

- `Steps` is the steps collected from the EntailmentBank.
The `Train_manual` and `Dev_manual` contain the annotated steps collected from the training and development split of EntailmentBank, respectively.
The `Train_pseudo` contains the steps with pseudo labels.
The `Dev_manual` is used to select the best checkpoint of entailment modules.

## Data for controller training
- `Controller_data` is the processed data used to train the controller.
We decompose the gold entailment trees into several intermediate states. We add disturbances to the trees to make positive and negative states based on our trained entailment modules.
One should follow the `code/make_controller_data.py` to make the controller training data based on their modules.

