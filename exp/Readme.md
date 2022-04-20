## Introduction

The trained models for `MetGen: A Module-based Entailment Tree Generation Framework for Answer Explanation`.

```
exp
├── Controller_task1                        # trained controller for Task 1
│   └── z7SOg44r
│       ├── config.json                     # experiment config
│       ├── model.config.json               # model config
│       ├── model_task1.pth                 # trained models
├── Controller_task2                        # trained controller for Task 2/3
│   └── NuCkQlfx
│       ├── config.json
│       ├── model.config.json
│       ├── model_task2.pth
└── Module_all                             
    ├── buffer_dict_para_etree_all.json     # buffer results for the module; used to speed up the reasoning process
    └── para_etree_all                      # trained prefixed entialment module
        └── Acdpaxg6
            ├── best_model.pth
            ├── config.json
            └── model.config.json

```