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
└── wiki_match                          # the synthetic data collected from wikipidia following ParaPattern(https://arxiv.org/abs/2104.08825)
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
