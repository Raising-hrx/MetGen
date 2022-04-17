from builtins import NotImplementedError
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
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import precision_score

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

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

from tree_utils import *
from sent_utils import add_fullstop, remove_fullstop
from evaluate_metric import *
from sent_utils import LCstring,sent_overlap

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



class controller_compare_sample_dataset(Dataset):

    def __init__(self, data_file, args):

        with open(data_file, 'r+') as f:
             datas = [json.loads(line) for line in f.readlines()]
                                
        if args.ratio_train < 1.0:
            ramiain_num = int(len(datas)*args.ratio_train)
            datas = datas[:ramiain_num]

        self.datas = datas
        self.args = args

        for data_item in datas:
            for key in ['states_positive_orig','states_positive_inter','states_positive_true_step','states_negative_replace',\
                       'states_negative_discard','states_negative_true_step','states_positive_true_step_abd',\
                       'states_negative_true_step_abd']:
                data_item['controller_data'][key] = self.modify_neg_fact_dtf(data_item['controller_data'][key])
                random.shuffle(data_item['controller_data'][key])
            
        for data_item in datas:
            compare_item_samples = []
                        
            # if 3 in args.compare_strategy:
            compare_pool_pos = data_item['controller_data']['states_positive_orig']
            compare_pool_pos += data_item['controller_data']['states_positive_inter']
            compare_pool_pos += data_item['controller_data']['states_positive_true_step']
            compare_pool_pos += data_item['controller_data']['states_positive_true_step_abd']

            compare_pool_neg = data_item['controller_data']['states_negative_replace']
            compare_pool_neg += data_item['controller_data']['states_negative_true_step']
            compare_pool_neg += data_item['controller_data']['states_negative_true_step_abd']

            for state1 in compare_pool_pos:
                if len(compare_item_samples) > 1000: continue
                if len(state1['pre']) > self.args.max_pre_training: continue

                states_compare_with_state1 = []
                for state2 in compare_pool_neg:
                    if len(state1['pre']) == len(state2['pre']):
                        states_compare_with_state1.append(state2)

                for state2 in states_compare_with_state1:
                    info_item1,info_item2 = self.make_compare_item(data_item, state1, state2, is_compare_fact=False)
                    if info_item1==None or info_item2==None: continue
                    compare_item_samples.append([info_item1,info_item2])
            # end if
            
            data_item['compare_item_samples'] = compare_item_samples
            # gather by number of facts
            data_item['compare_item_samples_gather'] = defaultdict(list)
            for info_item1,info_item2 in compare_item_samples:
                num_of_fact = len(info_item1['fact_ids'])
                data_item['compare_item_samples_gather'][num_of_fact].append([info_item1,info_item2])
            data_item['compare_item_samples_gather'] = dict(data_item['compare_item_samples_gather'])

                
        self.datas = [data_item for data_item in datas if len(data_item['compare_item_samples'])>0]
        

        print(f'loading controller data: {data_file}')
        print(f'task name: {args.task_name}')
        print(f'max_pre_training: {args.max_pre_training}')
        print(f'compare_strategy: {args.compare_strategy}')
        print(f'dataset length: {len(self.datas)}')
        print(f"all compare sample length: {sum([len(data_item['compare_item_samples']) for data_item in datas])}")
       
       
    def modify_neg_fact_dtf(self, state_list):
        neg_fact_dtf = 100
        for state in state_list:
            for sent_id in state['pre']:
                if state['fact_label'][sent_id] == 0: # negative fact
                    state['fact_dtf'][sent_id] = neg_fact_dtf
                    
        return state_list
    
    def make_compare_item(self, data_item, state1, state2, is_compare_fact=True):

        info_item1 = self.make_item(data_item, state1)
        info_item2 = self.make_item(data_item, state2)
        
        if info_item1==None or info_item2==None:
            return None, None
        
        # ------ compare fact between state1 & state2 -----
        
        compare_fact_between_info = []

        if is_compare_fact:
            compare_fact_by_dtf = []
            # within the state, the fact closer to the target (smaller depth_to_finsih dtf) should obtain a higger score
            # the distractor (neg fact with dtf=100) should have the lowest score
            for sent_1 in state1['pre']:
                for sent_2 in state2['pre']:
                    if state1['fact_dtf'][sent_1] < state2['fact_dtf'][sent_2]:
                        compare_fact_by_dtf.append([[state1['con'],sent_1],[state2['con'],sent_2], 1])
                    elif state1['fact_dtf'][sent_1] > state2['fact_dtf'][sent_2]:
                        compare_fact_by_dtf.append([[state1['con'],sent_1],[state2['con'],sent_2], -1])

            compare_fact_between_info = compare_fact_by_dtf


        compare_fact_between_idx1 = []
        compare_fact_between_idx2 = []
        compare_fact_between_label = []

        if is_compare_fact:
            for (target_id1, fact_id1), (target_id2, fact_id2), label in compare_fact_between_info:
                idx1 = info_item1['target_ids'].index(target_id1)*len(info_item1['fact_ids']) + info_item1['fact_ids'].index(fact_id1)
                idx2 = info_item2['target_ids'].index(target_id2)*len(info_item2['fact_ids']) + info_item2['fact_ids'].index(fact_id2)
                compare_fact_between_idx1.append(idx1)
                compare_fact_between_idx2.append(idx2)
                compare_fact_between_label.append(label)

        info_item1['compare_fact_between_idx1'] = compare_fact_between_idx1
        info_item2['compare_fact_between_idx2'] = compare_fact_between_idx2
        info_item1['compare_fact_between_label'] = compare_fact_between_label
        
        info_item1['compare_fact_between_info'] = compare_fact_between_info
        
        
        info_item1 = self.make_compare_within_item(info_item1, state1)
        info_item2 = self.make_compare_within_item(info_item2, state2)

        return info_item1, info_item2
        
    def make_compare_within_item(self, info_item, state):
        # ------ compare fact within each state -----
        compare_fact_within_info = []
        for sent_1, sent_2 in combinations(state['pre'],2):
            if state['fact_dtf'][sent_1] < state['fact_dtf'][sent_2]:
                compare_fact_within_info.append([[state['con'],sent_1],[state['con'],sent_2],1])
            elif state['fact_dtf'][sent_1] > state['fact_dtf'][sent_2]:
                compare_fact_within_info.append([[state['con'],sent_1],[state['con'],sent_2],-1])

        compare_fact_within_idx1 = []
        compare_fact_within_idx2 = []
        compare_fact_within_label = []
        for (target_id1, fact_id1), (target_id2, fact_id2), label in compare_fact_within_info:
            idx1 = info_item['target_ids'].index(target_id1)*len(info_item['fact_ids']) + info_item['fact_ids'].index(fact_id1)
            idx2 = info_item['target_ids'].index(target_id2)*len(info_item['fact_ids']) + info_item['fact_ids'].index(fact_id2)
            compare_fact_within_idx1.append(idx1)
            compare_fact_within_idx2.append(idx2)
            compare_fact_within_label.append(label)
        
        info_item['compare_fact_within'] = [
            compare_fact_within_idx1,
            compare_fact_within_idx2,
            compare_fact_within_label
        ]
        info_item['compare_fact_within_info'] = compare_fact_within_info
        
        # ----- compare step within each state -----
        all_steps = list(combinations(state['pre'],2))
        all_steps = [sorted(list(step)) for step in all_steps]
        
        compare_step_within_idx1 = []
        compare_step_within_idx2 = []
        compare_step_within_label = []
        compare_step_within_info = []

        for pos_step in info_item['positive_next_steps']:
            for neg_step in info_item['negative_next_steps']:
                idx1 = all_steps.index(sorted(pos_step))
                idx2 = all_steps.index(sorted(neg_step))
                compare_step_within_idx1.append(idx1)
                compare_step_within_idx2.append(idx2)
                compare_step_within_label.append(1)
                compare_step_within_info.append([pos_step,neg_step,1])
        
        info_item['compare_step_within'] = [
            compare_step_within_idx1,
            compare_step_within_idx2,
            compare_step_within_label
        ]
        info_item['compare_step_within_info'] = compare_step_within_info
        
        steps_label = [0]*len(all_steps)
        for pos_step in info_item['positive_next_steps']: 
            idx1 = all_steps.index(sorted(pos_step))
            steps_label[idx1] = 1
        info_item['steps_label'] = steps_label

        # ----- compare abduction step within each state -----
        all_steps_abd = [[state['con'], p_] for p_ in state['pre']]
        
        compare_step_abd_within_idx1 = []
        compare_step_abd_within_idx2 = []
        compare_step_abd_within_label = []
        compare_step_abd_within_info = []
        
        for pos_step in info_item['positive_next_steps_abd']:
            for neg_step in info_item['negative_next_steps_abd']:
                idx1 = all_steps_abd.index(pos_step)
                idx2 = all_steps_abd.index(neg_step)
                compare_step_abd_within_idx1.append(idx1)
                compare_step_abd_within_idx2.append(idx2)
                compare_step_abd_within_label.append(1)
                compare_step_abd_within_info.append([pos_step,neg_step,1])
        
        info_item['compare_step_abd_within'] = [
            compare_step_abd_within_idx1,
            compare_step_abd_within_idx2,
            compare_step_abd_within_label
        ]
        info_item['compare_step_abd_within_info'] = compare_step_abd_within_info
        
        steps_abd_label = [0]*len(all_steps_abd)
        for pos_step in info_item['positive_next_steps_abd']: 
            idx1 = all_steps_abd.index(pos_step)
            steps_abd_label[idx1] = 1
        info_item['steps_abd_label'] = steps_abd_label
        
        
        # state label 
        info_item['state_label']  = state['state_label']
        if info_item['state_label'] == 0:
            info_item['steps_label'] = []
            info_item['steps_abd_label'] = []
        
        return info_item
        
    def make_item(self, data_item, state):
        data_item_sents = data_item['controller_data']['all_sents']
        data_item_sents = {sent_idx:add_fullstop(sent) for sent_idx,sent in data_item_sents.items()}
        data_item['controller_data']['QA'] = add_fullstop(data_item['controller_data']['QA'])
        
        target_ids = []
        target_sents = []
        
        fact_ids = []
        fact_sents = []
        
        positive_next_steps = []
        negative_next_steps = []
        positive_next_steps_abd = []
        negative_next_steps_abd = []
        
        # -----load target sents-----

        # target: [H,QA,abd..]
        target_ids.append(state['con'])
        target_sents.append(data_item_sents[state['con']])
        if self.args.num_qa:
            target_ids.append('QA')
            target_sents.append(data_item['controller_data']['QA'])
        if self.args.num_abd and state['con'] == data_item['meta']['hypothesis_id']:

            state_used_sents = []
            for pre_, con_ in state['state_previous_steps']:
                state_used_sents += pre_
            
            add_abd_sents = []
            for sent_id in state['pre']:
                # assert sent_id not in state_used_sents
                if sent_id in state_used_sents: continue
                if sent_id not in data_item['controller_data']['abduction_H_sents']: continue # 没有反推结论
                abd_dict = data_item['controller_data']['abduction_H_sents'][sent_id]
                abd_sents = [abd_sent for abd_sent,abd_score in abd_dict.items()]
                abd_scores = [abd_score for abd_sent,abd_score in abd_dict.items()]

                max_index = np.argmax(abd_scores)
                max_score = abd_scores[max_index]
                max_sent = add_fullstop(abd_sents[max_index])
                if max_score < 0.28: continue
                
                new_id = f"{state['con']}-{sent_id}"
                add_abd_sents.append([new_id,max_sent,max_score])
         
            add_abd_sents = sorted(add_abd_sents,key=lambda x:x[2],reverse=True)
            for new_id, max_sent, max_score in add_abd_sents[:self.args.num_abd]:
                target_ids.append(new_id)
                target_sents.append(max_sent)
                data_item['controller_data']['all_sents'][new_id] = max_sent

        # -----load fact sents-----
        input_pre = state['pre']
        random.shuffle(input_pre)  # make random shuffle here; state['pre'] will be shuffle inplace

        sent2index = {}
        for i,sent_id in enumerate(input_pre):
            sent2index[sent_id] = i
            fact_ids.append(sent_id)
            fact_sents.append(data_item_sents[sent_id])
            

        # -----load steps-----
        if state['state_label'] == 1:
            positive_next_steps = state['gold_next_steps']
            negative_next_steps = state['false_next_steps']
            
            positive_next_steps_abd = state['gold_next_steps_abd']
            negative_next_steps_abd = [[state['con'],p_] for p_ in state['pre'] if [state['con'],p_] not in positive_next_steps_abd]

        # -----make score matric mask(for abduction relation)----- 
        # 1 for valid; 0 for not use
        fact_score_mask = []
        for target_id in target_ids:
            for fact_id in fact_ids:
                if fact_id in target_id: # "sent1" in "H-sent1"
                    fact_score_mask.append(0)
                else:
                    fact_score_mask.append(1)
            
        if len((" ".join(target_sents+fact_sents)).split()) > 300:
            # print('TOO LONG')
            return None
            
        info_item = {
            'target_ids':target_ids,
            'target_sents':target_sents,
            'fact_ids':fact_ids,
            'fact_sents':fact_sents,
            'positive_next_steps':positive_next_steps,
            'negative_next_steps':negative_next_steps,
            'fact_score_mask':fact_score_mask,
            'positive_next_steps_abd':positive_next_steps_abd,
            'negative_next_steps_abd':negative_next_steps_abd,

            'meta_state_info':state,
        }

        return info_item
    
    def __getitem__(self, index):
        data_item = self.datas[index]
        
        if vars(self.args).get('sample_strategy', 1) == 1:
            info_item1, info_item2 = random.choice(data_item['compare_item_samples'])
        elif self.args.sample_strategy == 2:
            num_of_fact = random.choice(list(data_item['compare_item_samples_gather'].keys()))
            info_item1, info_item2 = random.choice(data_item['compare_item_samples_gather'][num_of_fact])
        else:
            info_item1, info_item2 = random.choice(data_item['compare_item_samples'])
        
        return info_item1, info_item2
    
    def __len__(self):
        return len(self.datas)



class LinearsHead(nn.Module):
    def __init__(self, config, input_num=2):
        super(LinearsHead, self).__init__()
        self.linear1 = nn.Linear(input_num*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) # logit
        return x


class Controller(nn.Module):
    def __init__(self, args, config):
        super(Controller, self).__init__()
        
        self.args = args
        self.config = config
        
        # self.encoder = AutoModel.from_pretrained(args.model_name_or_path,config=config,local_files_only=True)
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path,config=config)
        self.sentsim = LinearsHead(config, input_num=2)
        self.stephead = LinearsHead(config, input_num=2)
        self.statehead = LinearsHead(config, input_num=1)
        self.stephead_abd = LinearsHead(config, input_num=2)

        self.device = self.encoder.device
        
        if self.args.state_method == 'fact_cls_learn':
            lambda_weight = torch.FloatTensor([0.5, 0.5])
            lambda_weight = nn.Parameter(lambda_weight,requires_grad=True)
            self.lambda_weight = lambda_weight
            assert self.lambda_weight.requires_grad == True
        

    def forward(self, batch_encoder_inputs, batch_label_info):
        raise NotImplementedError
        
    def score_instance(self, batch_encoder_inputs, batch_info):
        
        batch_size = batch_encoder_inputs['input_ids'].shape[0]
        outputs = self.encoder(**batch_encoder_inputs)
        
        cls_features = outputs['pooler_output']
        
        batch_pred_state_score = []
        batch_pred_fact_score = []
        batch_pred_step_score = []
        batch_pred_step_abd_score = []
        
        batch_pred_state_score_by_cls = self.statehead(cls_features).view(-1).sigmoid() # logit -> 0~1
        for batch_index in range(batch_size):   
            seq_features = outputs['last_hidden_state'][batch_index]
            info = batch_info[batch_index]
            
            # -----collect target / fact features-----
            target_features = []
            for to in info['target_offsets']:
                target_features.append(torch.mean(seq_features[to[0]:to[1],:], dim=0))
                
            fact_features = []
            for fo in info['fact_offsets']:
                fact_features.append(torch.mean(seq_features[fo[0]:fo[1],:], dim=0))
                
            # -----compute sent similarity-----
            fact_cls_features = []
            for target_feature in target_features:
                for fact_feature in fact_features:
                    fact_cls_features.append(torch.cat((fact_feature,target_feature)).unsqueeze(0)) # [1,1024*2]
  
            fact_cls_features = torch.cat(fact_cls_features, dim = 0) # [traget_num * facts_num, 1024*2]
            pred_fact_score = self.sentsim(fact_cls_features).view(-1)
            pred_fact_score = pred_fact_score.sigmoid() # logit -> 0~1

            # -----compute state score-----
            pred_fact_score_matrix = pred_fact_score.view(len(fact_features),len(target_features))
            fact_score_mask = torch.tensor(info['fact_score_mask']).view(len(fact_features),len(target_features)).to(self.device)
            max_score_per_fact, _ = torch.max(pred_fact_score_matrix * fact_score_mask,dim=1)
            if self.args.state_method == 'fact_cls_learn':
                pred_state_score = torch.mean(max_score_per_fact)
                pred_state_score_by_cls = batch_pred_state_score_by_cls[batch_index]
                weight = self.lambda_weight.softmax(dim=0)
                pred_state_score = weight[0] * pred_state_score + weight[1] * pred_state_score_by_cls
            else:
                raise NotImplementedError
                
                
            # ----- compute deduction step scores and abduction step scores -----   
            if len(fact_features) > 1:
                step_features = []
                target_side_feature = sum(target_features) / len(target_features) 
                for idx1,idx2 in list(combinations(range(len(fact_features)),2)):
                    # consider all combinations with 2 facts
                    step_features.append(torch.cat((fact_features[idx1],fact_features[idx2])).unsqueeze(0))  # [1,1024*2]
                step_features = torch.cat(step_features, dim = 0) # [len(steps), 1024*3]
                pred_step_score = self.stephead(step_features).view(-1)
                
                
                step_abd_features = []
                target_side_feature = sum(target_features) / len(target_features) 
                for idx in range(len(fact_features)):
                    # consider all combinations [H, fact]
                    step_abd_features.append(torch.cat((target_side_feature,fact_features[idx])).unsqueeze(0))  # [1,1024*2]
                step_abd_features = torch.cat(step_abd_features, dim = 0) # [len(facts), 1024*3]
                pred_step_abd_score = self.stephead_abd(step_abd_features).view(-1)
                
                if self.args.step_func == 'softmax_all':
                    pred_step_score_all = torch.cat([pred_step_score,pred_step_abd_score])
                    pred_step_score_all = pred_step_score_all.softmax(dim=0)
                    pred_step_score = pred_step_score_all[:len(pred_step_score)]
                    pred_step_abd_score = pred_step_score_all[len(pred_step_score):]
                else:
                    raise NotImplementedError
                    
            else:
                pred_step_score = None
                pred_step_abd_score = None
                

            batch_pred_state_score.append(pred_state_score)
            batch_pred_fact_score.append(pred_fact_score)
            batch_pred_step_score.append(pred_step_score)
            batch_pred_step_abd_score.append(pred_step_abd_score)
            
        return {
            'batch_pred_state_score':batch_pred_state_score,
            'batch_pred_fact_score':batch_pred_fact_score,
            'batch_pred_step_score':batch_pred_step_score,
            'batch_pred_step_abd_score':batch_pred_step_abd_score,
        }
        
    def training_pair(self, batch_encoder_inputs, batch_info):
        batch_size = batch_encoder_inputs['input_ids'].shape[0]
        score_results = self.score_instance(batch_encoder_inputs, batch_info)
        
        batch_pred_state_score = score_results['batch_pred_state_score']
        batch_pred_fact_score = score_results['batch_pred_fact_score']
        batch_pred_step_score = score_results['batch_pred_step_score']
        batch_pred_step_abd_score = score_results['batch_pred_step_abd_score']

        # ----- collect state compare loss -----
        batch_pred_state_score = torch.cat([score_.unsqueeze(0) for score_ in batch_pred_state_score])
        compare_state_x1 = batch_pred_state_score[range(0,batch_size,2)] # 2*k
        compare_state_x2 = batch_pred_state_score[range(1,batch_size,2)] # 2*k+1
        loss_state = F.margin_ranking_loss(compare_state_x1, compare_state_x2, 
                                            torch.ones_like(compare_state_x1).to(self.device),
                                            margin=self.args.margins[0])

        # ----- collect fact compare loss -----

        # compare fact between state
        loss_fact_between = []
        for k in range(batch_size//2):  # 2*k v.s. 2*k+1

            state1_compare_index = batch_info[2*k]['compare_fact_between_idx1']
            state2_compare_index = batch_info[2*k+1]['compare_fact_between_idx2']
            compare_label = batch_info[2*k]['compare_fact_between_label']
            
            pred_fact_score_state1 = batch_pred_fact_score[2*k]
            pred_fact_score_state2 = batch_pred_fact_score[2*k+1]
            
            if len(compare_label) == 0: continue
            compare_fact_x1 = pred_fact_score_state1[state1_compare_index]
            compare_fact_x2 = pred_fact_score_state2[state2_compare_index]
                        
            loss_ = F.margin_ranking_loss(compare_fact_x1, compare_fact_x2, torch.tensor(compare_label).to(self.device), margin=self.args.margins[1])
            loss_fact_between.append(loss_)
        
        # compare fact within state
        loss_fact_within = []
        for k in range(batch_size):
            compare_fact_within_idx1,compare_fact_within_idx2,compare_fact_within_label = batch_info[k]['compare_fact_within']
            pred_fact_score_state = batch_pred_fact_score[k]
            
            if len(compare_fact_within_label) == 0: continue
            compare_fact_x1 = pred_fact_score_state[compare_fact_within_idx1]
            compare_fact_x2 = pred_fact_score_state[compare_fact_within_idx2]
            loss_ = F.margin_ranking_loss(compare_fact_x1, compare_fact_x2, torch.tensor(compare_fact_within_label).to(self.device), margin=self.args.margins[1])
            loss_fact_within.append(loss_)
        
        loss_fact = loss_fact_between + loss_fact_within
        loss_fact = sum(loss_fact) / len(loss_fact) if len(loss_fact) else torch.tensor(0).to(self.device)
        
        # ----- collect step compare loss -----
        loss_step = []
        for k in range(batch_size):
            compare_step_within_idx1,compare_step_within_idx2,compare_step_within_label = batch_info[k]['compare_step_within']
            pred_step_score_state = batch_pred_step_score[k]
            
            if len(compare_step_within_label) == 0: continue
            if batch_info[k]['state_label'] == 0: continue
            compare_step_x1 = pred_step_score_state[compare_step_within_idx1]
            compare_step_x2 = pred_step_score_state[compare_step_within_idx2]
            loss_ = F.margin_ranking_loss(compare_step_x1, compare_step_x2, torch.tensor(compare_step_within_label).to(self.device), margin=self.args.margins[2])
            loss_step.append(loss_)
        loss_step = sum(loss_step) / len(loss_step) if len(loss_step) else torch.tensor(0).to(self.device)
        

        # ----- collect step abd compare loss -----
        loss_step_abd = []
        for k in range(batch_size):
            compare_step_abd_within_idx1,compare_step_abd_within_idx2,compare_step_abd_within_label = batch_info[k]['compare_step_abd_within']
            pred_step_abd_score_state = batch_pred_step_abd_score[k]
            
            if len(compare_step_abd_within_label) == 0: continue
            if batch_info[k]['state_label'] == 0: continue
            compare_step_x1 = pred_step_abd_score_state[compare_step_abd_within_idx1]
            compare_step_x2 = pred_step_abd_score_state[compare_step_abd_within_idx2]
            loss_ = F.margin_ranking_loss(compare_step_x1, compare_step_x2, torch.tensor(compare_step_abd_within_label).to(self.device), margin=self.args.margins[2])
            loss_step_abd.append(loss_)
        loss_step_abd = sum(loss_step_abd) / len(loss_step_abd) if len(loss_step_abd) else torch.tensor(0).to(self.device)
        
        # ------ classification loss ------
        # state
        batch_gold_state_labels = [info['state_label'] for info in batch_info]
        batch_gold_state_labels = torch.tensor(batch_gold_state_labels,dtype=torch.float).to(self.device)
        # add softmax bewteen comapre state
        batch_pred_state_score_softmax = batch_pred_state_score.view(-1,2).softmax(dim=1).view(-1)
        loss_state_cls = nn.BCELoss()(batch_pred_state_score_softmax, batch_gold_state_labels)        
        
        # step
        loss_step_cls = []
        for k in range(batch_size):
            gold_step_labels = batch_info[k]['steps_label']
            if len(gold_step_labels) == 0: continue
            if batch_info[k]['state_label'] == 0: continue
            if batch_pred_step_score[k] is None: continue
            gold_step_labels = torch.tensor(gold_step_labels,dtype=torch.float).to(self.device)
            loss_ = nn.BCELoss()(batch_pred_step_score[k], gold_step_labels) 
            loss_step_cls.append(loss_)
        loss_step_cls = sum(loss_step_cls) / len(loss_step_cls) if len(loss_step_cls) else torch.tensor(0).to(self.device)
        
        # step abd
        loss_step_abd_cls = []
        for k in range(batch_size):
            gold_step_abd_labels = batch_info[k]['steps_abd_label']
            if len(gold_step_abd_labels) == 0: continue
            if batch_info[k]['state_label'] == 0: continue
            if batch_pred_step_abd_score[k] is None: continue
            gold_step_abd_labels = torch.tensor(gold_step_abd_labels,dtype=torch.float).to(self.device)
            loss_ = nn.BCELoss()(batch_pred_step_abd_score[k], gold_step_abd_labels) 
            loss_step_abd_cls.append(loss_)
        loss_step_abd_cls = sum(loss_step_abd_cls) / len(loss_step_abd_cls) if len(loss_step_abd_cls) else torch.tensor(0).to(self.device)
        
        # fact
        loss_fact_cls = []
        for k in range(batch_size):
            info_item = batch_info[k]
            state = info_item['meta_state_info']
            
            fact_cls_index = []
            fact_cls_label = []
            for idx, sent_id in enumerate(info_item['fact_ids']):
                if state['fact_dtf'][sent_id] == 0:
                    fact_cls_index.append(idx)
                    fact_cls_label.append(1)
                if state['fact_dtf'][sent_id] == 100:
                    fact_cls_index.append(idx)
                    fact_cls_label.append(0)        
            if len(fact_cls_index)==0: continue
            pred_fact_cls_score = batch_pred_fact_score[k][fact_cls_index]
            fact_cls_label = torch.tensor(fact_cls_label,dtype=torch.float).to(self.device)
            
            loss_ = nn.BCELoss()(pred_fact_cls_score, fact_cls_label) 
            loss_fact_cls.append(loss_)
        loss_fact_cls = sum(loss_fact_cls) / len(loss_fact_cls) if len(loss_fact_cls) else torch.tensor(0).to(self.device)

        # merger loss
        if self.args.add_state_cls_loss > 0.0:
            loss_state = loss_state + self.args.add_state_cls_loss*loss_state_cls
        if self.args.add_fact_cls_loss > 0.0:
            loss_fact = loss_fact + self.args.add_fact_cls_loss*loss_fact_cls
        if self.args.add_step_cls_loss > 0.0:
            loss_step = loss_step + self.args.add_step_cls_loss*loss_step_cls
            loss_step_abd = loss_step_abd + self.args.add_step_cls_loss*loss_step_abd_cls

        return {
            'loss_state':loss_state,
            'loss_fact':loss_fact,
            'loss_step':loss_step,
            'loss_step_abd':loss_step_abd
        }



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


def tokenize_target_fact_sents(target_sents, fact_sents, tokenizer, pad_max_length = None):
    target_offsets = []
    fact_offsets = []
    
    input_ids = []
    attention_mask = []
    
    # make input ids
    input_ids.append(tokenizer.cls_token_id) 

    for s in target_sents:
        new_token = tokenizer.encode(s,add_special_tokens=False)
        new_offsets = [len(input_ids), len(input_ids)+len(new_token)]
        target_offsets.append(new_offsets)

        input_ids += new_token
        input_ids += [tokenizer.sep_token_id]

    input_ids += [tokenizer.sep_token_id]

    for s in fact_sents:
        new_token = tokenizer.encode(s,add_special_tokens=False)
        new_offsets = [len(input_ids), len(input_ids)+len(new_token)]
        fact_offsets.append(new_offsets)

        input_ids += new_token
        input_ids += [tokenizer.sep_token_id]

    attention_mask = [1] * len(input_ids)
    
    if pad_max_length:
        attention_mask += [0] * (pad_max_length - len(input_ids))
        input_ids += [tokenizer.pad_token_id] * (pad_max_length - len(input_ids))
    
    return input_ids, attention_mask, target_offsets, fact_offsets

def train_one_step(batch, model, tokenizer, args):

    model.train()
    
    # process batch data
    batch_encoder_inputs = {'input_ids':[],'attention_mask':[]}
    batch_label_info = []
    for item in batch:
        input_ids, attention_mask, target_offsets, fact_offsets = \
            tokenize_target_fact_sents(item['target_sents'], item['fact_sents'], tokenizer,pad_max_length=args.max_token_length)
        
        batch_encoder_inputs['input_ids'].append(input_ids)
        batch_encoder_inputs['attention_mask'].append(attention_mask)
        
        info_item = copy.deepcopy(item)
        info_item.update({
            'target_offsets':target_offsets,
            'fact_offsets':fact_offsets,
        })
        batch_label_info.append(info_item)
        
    batch_encoder_inputs['input_ids'] = torch.LongTensor(batch_encoder_inputs['input_ids']).to(model.device)
    batch_encoder_inputs['attention_mask'] = torch.LongTensor(batch_encoder_inputs['attention_mask']).to(model.device)

    model_return = model.training_pair(batch_encoder_inputs,batch_label_info)
    
    return model_return

def eval_model_by_loss(data_loader, model, tokenizer, args):

    model.eval()

    metric_collecter = defaultdict(list)

    with torch.no_grad():

        for batch in data_loader:
            
            # process batch data
            batch_encoder_inputs = {'input_ids':[],'attention_mask':[]}
            batch_label_info = []
            for item in batch:
                input_ids, attention_mask, target_offsets, fact_offsets = \
                    tokenize_target_fact_sents(item['target_sents'], item['fact_sents'], tokenizer,pad_max_length=args.max_token_length)
                
                batch_encoder_inputs['input_ids'].append(input_ids)
                batch_encoder_inputs['attention_mask'].append(attention_mask)
                
                info_item = copy.deepcopy(item)
                info_item.update({
                    'target_offsets':target_offsets,
                    'fact_offsets':fact_offsets,
                })
                batch_label_info.append(info_item)
            
            batch_encoder_inputs['input_ids'] = torch.LongTensor(batch_encoder_inputs['input_ids']).to(model.device)
            batch_encoder_inputs['attention_mask'] = torch.LongTensor(batch_encoder_inputs['attention_mask']).to(model.device)

            model_return = model.training_pair(batch_encoder_inputs,batch_label_info)

            for k,v in model_return.items():
                if 'loss' in k:
                    metric_collecter[k].append(v.detach().clone().cpu().data)
                elif 'labels' in k:
                    metric_collecter[k] += v

    for k,v in metric_collecter.items():
        if 'loss' in k:
            metric_collecter[k] = np.mean(v)

    return dict(metric_collecter)


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


    train_dataset = controller_compare_sample_dataset(args.train_data_file,args)
    dev_dataset = controller_compare_sample_dataset(args.dev_data_file,args)

    def collect_compare(batch):
        new_batch = []
        for group in batch:
            new_batch += group
        return new_batch

    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 4,
                            collate_fn = collect_compare)
    dev_loader = DataLoader(dataset = dev_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 4,
                            collate_fn = collect_compare)


    log.info(f"Length of training dataest: {len(train_dataset)}")
    log.info(f"Length of dev dataest: {len(dev_dataset)}")


    log.info(f"number of iteration each epoch : {len(train_loader)}")
    args.eval_iter = round(args.eval_epoch * len(train_loader))
    args.report_iter = round(args.report_epoch * len(train_loader))
    args.num_training_steps = args.epochs * len(train_loader)

    log.info("loading model")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # config = AutoConfig.from_pretrained(args.model_name_or_path,local_files_only=True)
    model = Controller(args,config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=False,local_files_only=True)

    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(state_dict)
        log.info(f"Resume model parameters form {args.resume_path}")

    model = model.to(device)

    model.device = model.encoder.device # 'Controller' object has no attribute 'device'


    with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(model.config), f, sort_keys=False, indent=4)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)
    
    log.info("start training")
    global_iter = 0
    loss_collecter = defaultdict(list)
    best_metric = -100

    for epoch_i in range(1, args.epochs+1):
        
        for batch in train_loader:
            model_return = train_one_step(batch,model,tokenizer,args)
            
            loss_state = model_return['loss_state']
            loss_fact = model_return['loss_fact']
            loss_step = model_return['loss_step']
            loss_step_abd = model_return['loss_step_abd']

            loss = args.loss_weight[0] * loss_state + \
                    args.loss_weight[1] * loss_fact + \
                    args.loss_weight[2] * loss_step + \
                    args.loss_weight[3] * loss_step_abd

            loss.backward()

            # add gradient clip
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            
            global_iter += 1
            
            
            if not global_iter % args.eval_iter:
                log.info(f"----- state evaluate -----")
                metric_collecter = eval_model_by_loss(dev_loader, model, tokenizer, args)
                # new_metric = (metric_collecter['acc_state'] + metric_collecter['acc_step'])/2

                metric_str = ""
                for k,v in metric_collecter.items():
                    if 'loss' in k or 'acc' in k:
                        metric_str += f" {k} {v:.4f}"

                log.info(f"Iteration {global_iter} dev {metric_str}")

                if args.save_model:
                    save_path = osp.join(args.exp_dir,f'model_{global_iter}.pth')
                    torch.save(model.state_dict(), save_path)
                    log.info(f"Iteration {global_iter} save model")

            if not global_iter % args.report_iter:

                show_str = ""
                for k,v in loss_collecter.items():
                    if 'loss' in k or 'acc' in k: 
                        show_str += f" {k} {np.mean(v):.4f} " 

                log.info(f"Epoch {global_iter/len(train_loader):.2f} training loss {show_str}")
                loss_collecter = defaultdict(list)


            else:
                for k,v in model_return.items():
                    if 'loss' in k:
                        loss_collecter[k].append(float(v.detach().clone().cpu().data))
                    elif 'labels' in k:
                        loss_collecter[k] += v


        log.info(f"Epoch {epoch_i} finished")


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training BART')

    # dateset
    parser.add_argument("--train_data_file", type=str, 
                        default='', help="training data file")
    parser.add_argument("--dev_data_file", type=str, 
                        default='', help="dev data file")
    parser.add_argument("--test_data_file", type=str, 
                        default='', help="test data file")     
    parser.add_argument("--data_loading_type", type=str, 
                        default='orig', help="test data file")  
    parser.add_argument("--compare_group_len", type=int, 
                        default=1, help="")  
    parser.add_argument("--max_pre_training", type=int, 
                        default=100, help="")  
    parser.add_argument("--abd_compare", action='store_true', default=False) 
    parser.add_argument('--compare_strategy', type=int, 
                        nargs='+',default=[3], help='compare_strategy')
    parser.add_argument("--sample_strategy", type=int, default=1, help="") 
    parser.add_argument('--ratio_train', type=float, default=1.0)

    
    # model
    parser.add_argument("--model_name_or_path", type=str, 
                        default="facebook/bart-large", help="")  
    parser.add_argument("--resume_path", type=str, 
                        default="", help="")       
    parser.add_argument("--task_name", type=str, 
                        default="", help="")   
    parser.add_argument("--num_qa", type=int, 
                        default=0, help="")    
    parser.add_argument("--num_abd", type=int, 
                        default=0, help="")   
    parser.add_argument("--state_method", type=str, 
                        default='fact_cls_learn', help="")  
    parser.add_argument("--num_max_fact", type=int, 
                        default=20, help="") 
    parser.add_argument("--max_token_length", type=int, 
                        default=450, help="")   
    parser.add_argument("--step_func", type=str, 
                        default='softmax_all', help="") 
                        
    # optimization
    parser.add_argument('--bs', type=int, default=5, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
                        
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--eval_epoch', type=float, default=1.0)

    parser.add_argument('--loss_weight', type=float, 
                        nargs='+',default=[1.0, 1.0, 1.0, 1.0], help='state_fact_step_stepabd')
    parser.add_argument('--add_state_cls_loss', type=float, default=0.0)
    parser.add_argument('--add_fact_cls_loss', type=float, default=1.0)
    parser.add_argument('--add_step_cls_loss', type=float, default=1.0)

    parser.add_argument('--margins', type=float, 
                        nargs='+',default=[0.1,0.1,0.0], help='state_fact_step_loss_margin')

    # seed
    parser.add_argument('--seed', type=int, default=1260, metavar='S',
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