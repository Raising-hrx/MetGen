import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import math
import time
import argparse

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BartForConditionalGeneration,BartTokenizer
from transformers import T5ForConditionalGeneration,T5Tokenizer

from transformers.optimization import Adafactor,AdamW,get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

import bleurt 
from bleurt import score
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


##### experiment utils
def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))
#####


def uncapitalize(sent):
    if len(sent) > 1:
        return sent[0].lower() + sent[1:]
    else:
        return sent.lower()
    
def uncapitalize(sent):
    if len(sent) > 1:
        return sent[0].lower() + sent[1:]
    else:
        return sent.lower()
    
class pattern_dataset_all(Dataset):


    def __init__(self, data_files, task_prefixes, loading_types):
    
        input_cap = False
        
        all_datas = {}
        for data_file, task_prefix, loading_type in zip(data_files, task_prefixes, loading_types):
        
            with open(data_file, 'r') as f:
                 data_items = [json.loads(line) for line in f.readlines()]

            all_sents = []
            for item in data_items:
                if loading_type == 'orig':
                    all_sents.append([item['input_1'],item['input_2'],item['output']])
                elif loading_type == 'orig2x':
                    all_sents.append([item['input_1'],item['input_2'],item['output']])
                    all_sents.append([item['input_2'],item['input_1'],item['output']])
                elif loading_type == 'para1x':
                    all_sents.append([item['input_1'],item['input_2'],item['output']])
                    all_sents.append([item['input_1_paras'][0],item['input_2_paras'][0],item['output']])
                elif loading_type == 'para2x':
                    all_sents.append([item['input_1'],item['input_2'],item['output']])
                    all_sents.append([item['input_1_paras'][0],item['input_2_paras'][0],item['output']])
                    all_sents.append([item['input_2_paras'][1],item['input_1_paras'][1],item['output']])
                elif loading_type == 'para_only':
                    all_sents.append([item['input_1_paras'][0],item['input_2_paras'][0],item['output_paras'][0]])
                    all_sents.append([item['input_2_paras'][1],item['input_1_paras'][1],item['output_paras'][1]])
                elif loading_type == 'orig_abduction':
                    all_sents.append([item['output'],item['input_1'],item['input_2']])
                    all_sents.append([item['output'],item['input_2'],item['input_1']])
                elif loading_type == 'para2x_abduction':
                    all_sents.append([item['output'],item['input_1'],item['input_2']])
                    all_sents.append([item['output'],item['input_2'],item['input_1']])
                    all_sents.append([item['output_paras'][0],item['input_1_paras'][0],item['input_2']])
                    all_sents.append([item['output_paras'][1],item['input_1_paras'][1],item['input_2']])
                    all_sents.append([item['output_paras'][0],item['input_2_paras'][0],item['input_1']])
                    all_sents.append([item['output_paras'][1],item['input_2_paras'][1],item['input_1']])
                else:
                    raise NotImplementedError 

            
            if input_cap:
                all_sents = self.capitalize_all(all_sents)
            else:
                all_sents = self.uncapitalize_all(all_sents)

            if task_prefix not in all_datas:
                all_datas[task_prefix] = all_sents
            else:
                all_datas[task_prefix] += all_sents
            
        self.all_datas = all_datas
            
        self.data_files = data_files
        self.task_prefix = task_prefix
        self.loading_types = loading_types
        
        self.input_cap = input_cap
        
        for k,v in all_datas.items():
            print(f"data of prefix {k}: {len(v)}")
            
        max_per_type = 300000
        self.all_sents_with_prefix = []
        for task_prefix, sents in all_datas.items():
            random.shuffle(sents)
            for ss in sents[:max_per_type]:
                self.all_sents_with_prefix.append([task_prefix]+ss)
            
            
        
        
        
    def __getitem__(self, index):
        return self.all_sents_with_prefix[index]
    
    def __len__(self):
        return len(self.all_sents_with_prefix)
    
    def capitalize_all(self, all_sents):
        tmp_sents = []
        for sents in all_sents:
            tmp_sents.append([s.capitalize() for s in sents])
        return tmp_sents
    
    def uncapitalize_all(self, all_sents):
        tmp_sents = []
        for sents in all_sents:
            tmp_sents.append([uncapitalize(s) for s in sents])
        return tmp_sents

class etree_step_dataset_all(Dataset):
    def __init__(self, data_files, task_prefixes, etree_loading_types, ratio_first=1.0):
        input_cap = False
        self.input_cap = input_cap
        
        all_datas = {}
        for data_file, task_prefix, loading_type in zip(data_files, task_prefixes, etree_loading_types):
        
            with open(data_file, 'r') as f:
                 data_items = [json.loads(line) for line in f.readlines()]

            if ratio_first < 1.0:
                remain_num = int(ratio_first * len(data_items))
                data_items = data_items[:remain_num]
                
            all_sents = []
            for item in data_items:
                # tmp[:-1] for input sentences; tmp[-1] for output sentences

                if loading_type == 'deduction':
                    tmp = []
                    for s in item['step_pre']:
                        input_sent = self.pre_process(item['triples'][s])
                        tmp.append(input_sent)

                    output_sent = self.pre_process(item['triples'][item['step_con']])
                    tmp.append(output_sent)
                    all_sents.append(tmp)

                elif loading_type == 'abduction': 

                    for s in item['step_pre']:
                        tmp = []

                        orig_output_sent = self.pre_process(item['triples'][item['step_con']])
                        tmp.append(orig_output_sent)

                        output_sent = self.pre_process(item['triples'][s])

                        for other_s in item['step_pre']:
                            if other_s != s:
                                input_sent = self.pre_process(item['triples'][other_s])
                                tmp.append(input_sent)

                        tmp.append(output_sent)
                        all_sents.append(tmp)

                else:
                    raise NotImplementedError

            if task_prefix not in all_datas:
                all_datas[task_prefix] = all_sents
            else:
                all_datas[task_prefix] += all_sents
        
        
        self.all_datas = all_datas
            
        self.data_files = data_files
        self.task_prefix = task_prefix
        self.loading_types = etree_loading_types
        
        for k,v in all_datas.items():
            print(f"data of prefix {k} {len(v)}")
            
        self.all_sents_with_prefix = []
        for task_prefix, sents in all_datas.items():
            random.shuffle(sents)
            for ss in sents:
                self.all_sents_with_prefix.append([task_prefix] + ss)
            
        
    def __getitem__(self, index):
        return self.all_sents_with_prefix[index]
    
    def __len__(self):
        return len(self.all_sents_with_prefix)
    
    def pre_process(self, sent):
        sent = sent + '.'
        if self.input_cap:
            sent = sent.capitalize()
        else:
            sent = uncapitalize(sent)
            
        return sent
    
    
def create_optimizer(model,args):
    # decay if not LayerNorm or bias
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps":1e-6,
        }
    optimizer_kwargs["lr"] = args.lr
    
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(optimizer, args):
    warmup_steps = math.ceil(args.num_training_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    return lr_scheduler

def train_one_step(batch, model, tokenizer, args):
    r"""
    train the model one step with the given batch data
    return the loss
    """
    model.train()
    
    # process batch data
    input_sents = [item[:-1] for item in batch]
    output_sents = [item[-1] for item in batch]

    # join for  "<s> sent1 sent2 </s>" else "<s> sent1 </s></s> sent2 </s>"  
    if args.input_join:
        input_sents = [' '.join(sents) for sents in input_sents]  
    else:
        input_sents = [tokenizer.sep_token.join(sents) for sents in input_sents]  

    input_batch = tokenizer(
            input_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length', # 'longest',
            max_length=args.max_src_length,
            truncation=True,)

    output_batch = tokenizer(
                output_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding= 'max_length', # 'longest',
                max_length=args.max_tgt_length,
                truncation=True,)

    label_batch = output_batch['input_ids']
    label_batch.masked_fill_(label_batch == tokenizer.pad_token_id, -100) 
    
    input_batch['labels'] = label_batch
    input_batch = input_batch.to(model.device)
    
    # forward
    model_return = model(**input_batch)

    return model_return['loss']

def eval_model(model,data_loader, tokenizer, bleurt_scorer=None):
    model.eval()

    inputs = []
    references = []
    candidates = []
    
    refer_ppls = []
    repeat_ppls = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for batch in data_loader:

        # process batch data
        input_sents = [item[:-1] for item in batch]
        output_sents = [item[-1] for item in batch]
        # join for  "<s> sent1 sent2 </s>" else "<s> sent1 </s></s> sent2 </s>"  
        if args.input_join:
            input_sents = [' '.join(sents) for sents in input_sents]   
        else:
            input_sents = [tokenizer.sep_token.join(sents) for sents in input_sents]  
        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=args.max_src_length,
                truncation=True,
            )
        input_batch = input_batch.to(model.device)
        
        # generate
        generated = model.generate(
            input_ids = input_batch['input_ids'],
            attention_mask = input_batch['attention_mask'],
            top_p = 0.9,
            do_sample = True,
            max_length= 50, 
            num_return_sequences = 1,  # can eval with more than one sample
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        inputs += input_sents
        references += output_sents
        candidates += decoded
        
        # refer PPL
        output_batch = tokenizer(
            output_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding= 'longest', 
            max_length=args.max_tgt_length,
            truncation=True,)
        input_batch['labels'] = output_batch['input_ids'].to(model.device)
        logits = model(**input_batch, use_cache=False)[1]
        loss = loss_fn(logits.view(-1, logits.shape[-1]), input_batch['labels'].view(-1))
        refer_ppls.append(loss.item())
        del logits
        
        # repeat PPL
        input_batch['labels'] = input_batch['input_ids'][:,1:].to(model.device).contiguous()
        logits = model(**input_batch, use_cache=False)[1]
        loss = loss_fn(logits.view(-1, logits.shape[-1]), input_batch['labels'].view(-1))
        repeat_ppls.append(loss.item())
        del logits
        
    # eval BLEURT
    if bleurt_scorer:
        bleurt_scores = bleurt_scorer.score(references = references, candidates=candidates)
    else:
        bleurt_scores = -1

    acc = np.sum(np.array(bleurt_scores) > 0.28) / len(bleurt_scores)

    return acc, np.mean(bleurt_scores), [inputs, candidates, references, bleurt_scores], np.exp(np.mean(refer_ppls)), np.exp(np.mean(repeat_ppls))

def eval_model_batch(model,batch, tokenizer, bleurt_scorer=None):
    model.eval()

    inputs = []
    references = []
    candidates = []

    # process batch data
    input_sents = [item[:-1] for item in batch]
    output_sents = [item[-1] for item in batch]
    # join for  "<s> sent1 sent2 </s>" else "<s> sent1 </s></s> sent2 </s>"  
    if args.input_join:
        input_sents = [' '.join(sents) for sents in input_sents]   
    else:
        input_sents = [tokenizer.sep_token.join(sents) for sents in input_sents]  
    input_batch = tokenizer(
            input_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            max_length=args.max_src_length,
            truncation=True,
        )
    input_batch = input_batch.to(model.device)
    
    # generate
    generated = model.generate(
        input_ids = input_batch['input_ids'],
        attention_mask = input_batch['attention_mask'],
        top_p = 0.9,
        do_sample = True,
        max_length= 50, 
        num_return_sequences = 1,  # can eval with more than one sample
    )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    inputs += input_sents
    references += output_sents
    candidates += decoded
        
    # eval
    if bleurt_scorer:
        bleurt_scores = bleurt_scorer.score(references = references, candidates=candidates)
    else:
        bleurt_scores = -1

    return np.mean(bleurt_scores), [inputs, candidates, references, bleurt_scores]


def run(args):

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.device_count() > 1:
    #     torch.cuda.manual_seed_all(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading data")

    if args.module_training_type =='para_all':
        train_data_files = [
            '../data/wiki_match/V1/Substitution/train.jsonl',
            '../data/wiki_match/V1/Conjunction/train.jsonl',
            '../data/wiki_match/V1/Ifthen/train.jsonl',
            '../data/wiki_match/V1/Substitution/train.jsonl',
            '../data/wiki_match/V1/Conjunction/train.jsonl',
            '../data/wiki_match/V1/Ifthen/train.jsonl',
        ]
        task_prefixes = [
            'deductive substitution:',
            'deductive conjunction:',
            'deductive if-then:',
            'abductive substitution:',
            'abductive conjunction:',
            'abductive if-then:',
        ]
        loading_types = [
            'para2x',
            'para2x',
            'para2x',
            'para2x_abduction',
            'para2x_abduction',
            'para2x_abduction',
        ]
        train_dataset = pattern_dataset_all(train_data_files,task_prefixes,loading_types)

        test_data_files = [
            '../data/wiki_match/V1/Substitution/test.jsonl',
            '../data/wiki_match/V1/Conjunction/test.jsonl',
            '../data/wiki_match/V1/Ifthen/test.jsonl',
            '../data/wiki_match/V1/Substitution/test.jsonl',
            '../data/wiki_match/V1/Conjunction/test.jsonl',
            '../data/wiki_match/V1/Ifthen/test.jsonl',
        ]
        task_prefixes = [
            'deductive substitution:',
            'deductive conjunction:',
            'deductive if-then:',
            'abductive substitution:',
            'abductive conjunction:',
            'abductive if-then:',
        ]
        loading_types = [
            'deduction',
            'deduction',
            'deduction',
            'abduction',
            'abduction',
            'abduction',
        ]
        test_dataset = etree_step_dataset_all(test_data_files,task_prefixes,loading_types)
        dev_dataset = test_dataset

    elif args.module_training_type =='etree_all':
        train_data_files = [
            '../data/Steps/Train_pseudo/pseudo.substitution.jsonl',
            '../data/Steps/Train_pseudo/pseudo.conjunction.jsonl',
            '../data/Steps/Train_pseudo/pseudo.if-then.jsonl',
            '../data/Steps/Train_pseudo/pseudo.substitution.jsonl',
            '../data/Steps/Train_pseudo/pseudo.conjunction.jsonl',
            '../data/Steps/Train_pseudo/pseudo.if-then.jsonl'
        ]
        task_prefixes = [
            'deductive substitution:',
            'deductive conjunction:',
            'deductive if-then:',
            'abductive substitution:',
            'abductive conjunction:',
            'abductive if-then:',
        ]
        loading_types = [
            'deduction',
            'deduction',
            'deduction',
            'abduction',
            'abduction',
            'abduction',
        ]
        train_dataset = etree_step_dataset_all(train_data_files,task_prefixes,loading_types,
                                                ratio_first=args.ratio_train)

        test_data_files = [
            '../data/wiki_match/V1/Substitution/test.jsonl',
            '../data/wiki_match/V1/Conjunction/test.jsonl',
            '../data/wiki_match/V1/Ifthen/test.jsonl',
            '../data/wiki_match/V1/Substitution/test.jsonl',
            '../data/wiki_match/V1/Conjunction/test.jsonl',
            '../data/wiki_match/V1/Ifthen/test.jsonl',
        ]
        task_prefixes = [
            'deductive substitution:',
            'deductive conjunction:',
            'deductive if-then:',
            'abductive substitution:',
            'abductive conjunction:',
            'abductive if-then:',
        ]
        loading_types = [
            'deduction',
            'deduction',
            'deduction',
            'abduction',
            'abduction',
            'abduction',
        ]
        test_dataset = etree_step_dataset_all(test_data_files,task_prefixes,loading_types)
        dev_dataset = test_dataset

    else:
        raise NotImplemented

    print("train_data_files")
    print(train_data_files)
    print("test_data_files")
    print(test_data_files)


    log.info(f"Length of training dataest: {len(train_dataset)}")
    log.info(f"Length of dev dataest: {len(dev_dataset)}")
    log.info(f"Length of test dataest: {len(test_dataset)}")

    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)
    dev_loader = DataLoader(dataset = dev_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = args.bs,
                            shuffle = False,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)

    log.info(f"number of iteration each epoch : {len(train_loader)}")
    args.eval_iter = round(args.eval_epoch * len(train_loader))
    args.report_iter = round(args.report_epoch * len(train_loader))
    args.num_training_steps = args.epochs * len(train_loader)


    log.info("loading model")
    if args.model_name_or_path in ['facebook/bart-large']:
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        model.config.update({
        'max_length': 142,
        'min_length': 8,
        'prefix': '',
        'decoder_start_token_id': tokenizer.bos_token_id
        })
    elif args.model_name_or_path in ['t5-large','t5-base','t5-small']:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model.config.update({
        'max_length': 142,
        'min_length': 8,
        })
        tokenizer.sep_token = tokenizer.eos_token
    else:
        raise NotImplementedError
    
    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(state_dict)
        log.info(f"Resume model parameters form {args.resume_path}")

    model = model.to(device)

    with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(model.config), f, sort_keys=False, indent=4)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)
    
    log.info("loading BLEURT")
    bleurt_scorer = score.BleurtScorer(args.bleurt_path) 

    log.info("start training")
    global_iter = 0
    loss_list = []
    best_metric = -100

    for epoch_i in range(1, args.epochs+1):
        
        for batch in train_loader:
            loss = train_one_step(batch,model,tokenizer,args)
            
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            
            global_iter += 1
            
            
            if not global_iter % args.eval_iter:
                eval_acc, eval_score, eval_info, refer_ppl, repeat_ppl = eval_model(model,test_loader,tokenizer,bleurt_scorer)
                inputs, candidates, references, scores = eval_info

                new_metric = eval_acc

                log.info(f"Iteration {global_iter} test acc: {eval_acc:.4f} bleurt socre:{eval_score:.4f} refer_ppl:{refer_ppl:.4f} repeat_ppl:{repeat_ppl:.4f}")
                with open(osp.join(args.exp_dir, f'prediction_{global_iter}.txt'), 'w') as f: 
                    f.write("----------Tesing set eval----------\n")
                    f.write(f"BLEURT score: {eval_score}\n")
                    for i_,c_,r_ ,s_ in zip(inputs, candidates, references, scores):
                        f.write(f"input: {i_}\n")
                        f.write(f"pred: {c_}\n")
                        f.write(f"refer: {r_}\n")
                        f.write(f"score: {s_}\n\n")

                eval_score, eval_info = eval_model_batch(model,batch,tokenizer,bleurt_scorer)
                inputs, candidates, references, scores = eval_info
                log.info(f"Iteration {global_iter} training batch bleurt socre {eval_score:.4f}")
                
                with open(osp.join(args.exp_dir, f'prediction_{global_iter}.txt'), 'a') as f: 
                    f.write("----------Training batch eval----------\n")
                    f.write(f"BLEURT score: {eval_score}\n")
                    for i_,c_,r_ ,s_ in zip(inputs, candidates, references, scores):
                        f.write(f"input: {i_}\n")
                        f.write(f"pred: {c_}\n")
                        f.write(f"refer: {r_}\n")
                        f.write(f"score: {s_}\n\n")

                if best_metric < new_metric:
                    best_metric = new_metric
                    log.info(f"------Iteration {global_iter} get best metric {best_metric:.4f}------")
                    if args.save_model:
                        save_path = osp.join(args.exp_dir,'best_model.pth')
                        torch.save(model.state_dict(), save_path)
                        log.info(f"Iteration {global_iter} save best model")

            if not global_iter % args.report_iter:
                log.info(f"Epoch {global_iter/len(train_loader):.1f} training loss {np.mean(loss_list):.4f}")
                loss_list = []
            else:
                loss_list.append(float(loss.cpu().data))
                
        log.info(f"Epoch {epoch_i} finished")

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training BART')

    parser.add_argument("--bleurt_path", type=str, default="../../bleurt/bleurt-large-512")  

    # dateset
    parser.add_argument("--module_training_type", type=str)  
    parser.add_argument('--ratio_train', type=float, default=1.0)
    parser.add_argument('--input_join', action='store_true', default=False)
    
    # model
    parser.add_argument("--model_name_or_path", type=str, 
                        default="facebook/bart-large", help="")  
    parser.add_argument("--resume_path", type=str, 
                        default="", help="")                
    parser.add_argument('--max_src_length', type=int, default=142, )
    parser.add_argument('--max_tgt_length', type=int, default=142, )

    # optimization
    parser.add_argument('--bs', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train')
                        
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--eval_epoch', type=float, default=1.0)

    # seed
    parser.add_argument('--seed', type=int, default=9731, metavar='S',
                        help='random seed')

    # exp and log
    parser.add_argument("--exp_dir", type=str, default='./exp')
    parser.add_argument("--code_dir", type=str, default='./code')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--report_epoch', type=float, default=1.0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_params()
    if args.seed == 0:
        args.seed = random.randint(1,1e4)

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())

    os.makedirs(args.exp_dir, exist_ok=True)
    
    # make metrics.json for logging metrics
    args.metric_file = osp.join(args.exp_dir, 'metrics.json')
    open(args.metric_file, 'a').close()

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    os.system(f'cp -r {args.code_dir} {args.exp_dir}')

    log.info('Python info: {}'.format(os.popen('which python').read().strip()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    run(args)

    # make 'done' file
    open(osp.join(args.exp_dir, 'done'), 'a').close()