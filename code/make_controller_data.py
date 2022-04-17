import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import argparse
from pprint import pprint
from itertools import combinations
import copy
from tqdm import tqdm

import numpy as np

from tree_utils import *
from evaluate_metric import *
from sent_utils import LCstring,sent_overlap,add_fullstop


###############################
# define args
task_name = "task_2"
reasoning_modules_type = "t5"
version_name = "v36"

device = 'cuda'
bleurt_path = "<path to Bleurt>"
exp_dir_single = "<path to module>"
###############################

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

import bleurt 
from bleurt import score
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
print("loading bleurt")
bleurt_scorer = score.BleurtScorer(bleurt_path) 


from reasoning import load_reasoning_module, module_generate, inference_alltype_batch


print("loading single module for deductive and abductive reasoning")
module_all,tokenizer_module,args_module = load_reasoning_module(exp_dir_single)
reasoning_modules = {
    'substitution':{
        'model':module_all.to(device),
        'task_prefix':'deductive substitution:',
    },
    'conjunction': {
        'model':module_all.to(device),
        'task_prefix':'deductive conjunction:',
    },
    'if-then':{
        'model':module_all.to(device),
        'task_prefix':'deductive if-then:',
    },
}

# abduction module
tokenizer_module_abd,args_module_abd = tokenizer_module,args_module
reasoning_modules_abd = {
    'substitution_abd':{
        'model':module_all.to(device),
        'task_prefix':'abductive substitution:',
    },
    'conjunction_abd': {
        'model':module_all.to(device),
        'task_prefix':'abductive conjunction:',
    },
    'if-then_abd':{
        'model':module_all.to(device),
        'task_prefix':'abductive if-then:',
    },
}


def collect_states_positive(input_tree, show = False):
    input_tree = input_tree
    input_steps = [[sorted(node['pre']),node['id']] for node in input_tree if node['pre']]
    
    print_node_tree(input_tree[0]['id'],input_tree) if show else None
    print("="*20) if show else None
    
    # add depth_to_finish
    def add_depth_to_finish(idx, node_list, dtf = 0):
        node = get_node(idx,node_list)
        node['depth_to_finish'] = dtf
        for pre_idx in node['pre']:
            add_depth_to_finish(pre_idx, node_list, dtf+1)

    add_depth_to_finish(input_tree[0]['id'],input_tree)

    # collect all intermediate state
    hypot_idx = input_tree[0]['id']
    all_states = []

    for n in range(1,len(input_steps)+1):
        for selected_steps in list(combinations(input_steps, n)):
            tmp_node_ids = set()
            for step_pre, step_con in selected_steps:
                for pre_idx in step_pre:
                    tmp_node_ids.add(pre_idx)
                tmp_node_ids.add(step_con)

            if hypot_idx not in tmp_node_ids:
                continue

            # verify whether the selected steps can make a tree
            tmp_tree = []
            for idx in tmp_node_ids:
                node = copy.deepcopy(get_node(idx,input_tree))
                if node['pre'] == []:
                    tmp_tree.append(node)
                elif all([pre_idx in tmp_node_ids for pre_idx in node['pre']]):
                    tmp_tree.append(node)
                else:
                    node['pre'] = []
                    tmp_tree.append(node)


            tmp_tree = get_tree(hypot_idx,tmp_tree)

            depth_to_finish = max([node['depth_to_finish'] for node in tmp_tree])
            length_to_finish = len([[sorted(node['pre']),node['id']] for node in tmp_tree if node['pre']])

            if tmp_tree not in all_states:
                all_states.append(tmp_tree)


    states_positive = []
    for state_tree in all_states:
        
        print_node_tree(state_tree[0]['id'],state_tree) if show else None
        
        pre = sorted(get_leaves_ids(state_tree[0]['id'],state_tree))
        con = state_tree[0]['id'] 
        depth_to_finish = max([node['depth_to_finish'] for node in state_tree])
        
        # future steps
        state_future_steps = [[sorted(node['pre']),node['id']] for node in state_tree if node['pre']]
        length_to_finish = len(state_future_steps)
        
        # previous steps
        state_previous_steps = []
        for step in input_steps:
            if step not in state_future_steps:
                state_previous_steps.append(step)

        gold_next_steps = []
        for step_pre, step_con in state_future_steps:
            if all([idx in pre for idx in step_pre]):
                # process the step to make it 2-premise
                if len(step_pre) == 1:
                    continue
                if len(step_pre) == 2:
                    gold_next_steps.append(step_pre)
                if len(step_pre) > 2:
                    for new_pre in list(combinations(step_pre, 2)):
                        sent1 = get_node(new_pre[0], state_tree)['sent']
                        sent2 = get_node(new_pre[1], state_tree)['sent']
                        
                        if sent_overlap(sent1,sent2,spacy_nlp,thre=0.75):
                            gold_next_steps.append(list(new_pre))
                            
        gold_next_steps = [sorted(step) for step in gold_next_steps]
                            
        false_next_steps = []
        for step in combinations(pre, 2):
            step = list(sorted(step))
            if step not in gold_next_steps:
                false_next_steps.append(step)
        
        # for task 1, we identify prove process by the number of pre sents
        length_to_finish = len(pre)-1 
        
        gold_next_steps_abd = []
        step_to_con = [future_step[0] for future_step in state_future_steps if future_step[1]==con][0]
        for p_ in pre:
            if p_ in step_to_con:
                gold_next_steps_abd.append([con, p_])
            
        
        
        states_positive.append({
            'pre':sorted(pre),
            'con':con,
            'depth_to_finish':depth_to_finish,
            'length_to_finish':length_to_finish,
            'state_future_steps':state_future_steps,
            'state_previous_steps':state_previous_steps,
            'state_label': 1,
            'fact_label': {idx:1 for idx in pre},
            'fact_dtf':{node['id']:node['depth_to_finish'] for node in state_tree}, # fact depth to finish
            'gold_next_steps':[sorted(step) for step in gold_next_steps],
            'false_next_steps':[sorted(step) for step in false_next_steps],
            'gold_next_steps_abd':gold_next_steps_abd,
            
        
        })

    states_positive.append({
        'pre':[hypot_idx],
        'con':hypot_idx,
        'depth_to_finish':0,
        'length_to_finish':0,
        'state_future_steps':[],
        'state_previous_steps':input_steps,
        'state_label': 1,
        'fact_label': {hypot_idx:1},
        'fact_dtf':{hypot_idx:0}, # fact depth to finish
        'gold_next_steps':[],
        'false_next_steps':[],
        'gold_next_steps_abd':[],
    })
    
    return states_positive


with open(f"../data/entailment_trees_emnlp2021_data_v2/dataset/{task_name}/train.jsonl",'r') as f:
    train_data = [json.loads(line) for line in f.readlines()]
with open(f"../data/entailment_trees_emnlp2021_data_v2/dataset/{task_name}/dev.jsonl",'r') as f:
    dev_data = [json.loads(line) for line in f.readlines()]


data_list = [train_data,dev_data]
name_list = [
    f'../data/Controller_data/train.controller.{task_name}.{version_name}.jsonl',
    f'../data/Controller_data/dev.controller.{task_name}.{version_name}.jsonl',
]


for process_data, save_name in zip(data_list,name_list):   
    print(f"making {save_name} len {len(process_data)}")

    print("Tree -> positive state")
    for data_item in tqdm(process_data):
        gt_tree = get_gt_node_list(data_item) 
        gt_steps = [[sorted(node['pre']),node['id']] for node in gt_tree if node['pre']]

        data_item['controller_data'] = {
            'hypothesis': data_item['hypothesis'],
            'QA': data_item['question'] + ' ' + data_item['answer'],
            'all_sents': {node['id']:node['sent'] for node in gt_tree},
            'states_positive_orig' : [],
            'states_positive_inter': [],
        }

            
        if len(gt_steps) > 15:
            print(f"Too long with {len(gt_steps)} steps")
            continue
            
        ### ----- make positive state whose target is H -----
        states_positive_orig = collect_states_positive(gt_tree)
        data_item['controller_data']['states_positive_orig'] = states_positive_orig

        ### ----- make positive state whose target is intermeidate node -----
        states_positive_inter = []
        inter_node_ids = [node['id'] for node in gt_tree[1:] if not node['id'].startswith('sent')]
        for idx in inter_node_ids:
            inter_tree = copy.deepcopy(get_tree(idx,gt_tree))
            states_positive_inter += collect_states_positive(inter_tree)
        data_item['controller_data']['states_positive_inter'] = states_positive_inter


    print("postive state -- replacement -> negative state")
    for data_item in tqdm(process_data):
        candidate_positive_states = data_item['controller_data']['states_positive_orig']
        candidate_positive_states += data_item['controller_data']['states_positive_inter']
        
        error_sents_from_delete_list = [i['fact'] for i in data_item['meta']['delete_list']]
        
        add_neg_sents = {}
        states_negative_replace = []
        
        for state in candidate_positive_states:
            neg_state = copy.deepcopy(state)
            neg_state['state_label'] = 0
        
            important_sent_ids = []
            for step in neg_state['gold_next_steps']:
                important_sent_ids += step
            if not important_sent_ids:
                important_sent_ids = neg_state['pre']
                
            not_important_sent_ids = [idx for idx in neg_state['pre'] if idx not in important_sent_ids]
                
            for selected_idx in important_sent_ids:
                selected_neg_sents = random.choice(error_sents_from_delete_list)
                neg_idx = f"neg_replace{len(add_neg_sents)}"
                add_neg_sents[neg_idx] = selected_neg_sents
                
                new_pre = []
                new_fact_label = {}
                for idx in neg_state['pre']:
                    if idx == selected_idx:
                        new_pre.append(neg_idx)
                        new_fact_label[neg_idx] = 0
                    else:
                        new_pre.append(idx)
                        new_fact_label[idx] = 1
                
            neg_state['state_label'] = 0
            neg_state['pre'] = new_pre
            neg_state['fact_label'] = new_fact_label
            neg_state['fact_dtf'][neg_idx] = 100
                
            states_negative_replace.append(neg_state)
                
        data_item['controller_data']['all_sents'].update(add_neg_sents)
        data_item['controller_data']['states_negative_replace'] = states_negative_replace

    print("postive state -- discard -> negative state")
    for data_item in tqdm(process_data):
        candidate_positive_states = data_item['controller_data']['states_positive_orig']
        candidate_positive_states += data_item['controller_data']['states_positive_inter']
        
        states_negative_discard = []
        
        for state in candidate_positive_states:
            neg_state = copy.deepcopy(state)
            neg_state['state_label'] = 0
        
            important_sent_ids = []
            for step in neg_state['gold_next_steps']:
                important_sent_ids += step
            if not important_sent_ids:
                important_sent_ids = neg_state['pre']
                
            not_important_sent_ids = [idx for idx in neg_state['pre'] if idx not in important_sent_ids]
                
            for selected_idx in important_sent_ids: 
                new_pre = []
                new_fact_label = {}
                for idx in neg_state['pre']:
                    if idx == selected_idx:
                        pass
                    else:
                        new_pre.append(idx)
                        new_fact_label[idx] = 1
                
            neg_state['state_label'] = 0
            neg_state['pre'] = new_pre
            neg_state['fact_label'] = new_fact_label
                
            if len(neg_state['pre']) == 0: continue
            states_negative_discard.append(neg_state)
                
        data_item['controller_data']['states_negative_discard'] = states_negative_discard

    print("positive state&step -- module --> postive/negative state")
    # compare with gold intermediate sent, bleurt>0.28 and max is positive, bleurt<0.28 and min is negative

    max_state_pre_data_item = 20

    pos_bleurt_thre = 0.28
    neg_bleurt_thre = 0.28

    if reasoning_modules_type == 'bart':
        buffer_name="infered_buffer.json" # bart
    elif reasoning_modules_type == 't5':
        buffer_name="infered_buffer_t5.json" #t5

    try:
        infered_buffer = json.load(open(buffer_name,'r'))
    except:
        infered_buffer = {}


    for data_item in tqdm(process_data):
        add_sents = {}
        states_positive_true_step = []
        states_negative_true_step = []
        
        
        candidate_states = data_item['controller_data']['states_positive_orig']
        random.shuffle(candidate_states)
        
        for state in candidate_states[:max_state_pre_data_item]:
            now_state = copy.deepcopy(state)
            
            now_state['gold_next_steps'] = [sorted(step) for step in now_state['gold_next_steps']]
            now_state_previous_steps_con = [step_con for step_pre,step_con in now_state['state_previous_steps']]
            
            
            for pre_idx in state['pre']:
                if pre_idx not in now_state_previous_steps_con:
                    continue 
                
                step_index = now_state_previous_steps_con.index(pre_idx)
                selected_step_pre, selected_step_con = now_state['state_previous_steps'][step_index]
                if len(selected_step_pre) != 2:
                    continue
                    
                assert selected_step_con == pre_idx
                
                
                sent1 = data_item['controller_data']['all_sents'][selected_step_pre[0]]
                sent2 = data_item['controller_data']['all_sents'][selected_step_pre[1]]
        
                buffer_key = sent1+ '+' + sent2
                if buffer_key not in infered_buffer:
                    input_sents = [[add_fullstop(sent1), add_fullstop(sent2)]]
                    output_sents, output_types = inference_alltype_batch(input_sents,(reasoning_modules,tokenizer_module,args_module))
                    output_sents, output_types = output_sents[0], output_types[0]
                
                    infered_buffer[buffer_key] = (output_sents, output_types)
                else:
                    output_sents, output_types = infered_buffer[buffer_key]
                    
                gold_inter_sents = data_item['controller_data']['all_sents'][selected_step_con]
                
                inter_scores = bleurt_scorer.score(references = [gold_inter_sents]*len(output_sents), candidates = output_sents)
                
                for sent_, score_ in zip(output_sents, inter_scores):
                    
                    if score_ == max(inter_scores) and score_ > pos_bleurt_thre:
                        
                        pos_idx = f"module_pos_int{len(add_sents)}"
                        add_sents[pos_idx] = sent_

                        new_pre = []
                        new_fact_label = {}

                        for idx in now_state['pre']:
                            if idx != selected_step_con:
                                new_pre.append(idx)
                                new_fact_label[idx] = 1

                        new_pre.append(pos_idx)
                        new_fact_label[pos_idx] = 1 # positive fact
                        
                        new_fact_dtf = copy.deepcopy(now_state['fact_dtf'])
                        new_fact_dtf[pos_idx] = now_state['fact_dtf'][selected_step_con]

                        new_gold_next_steps = []
                        for step_ in now_state['gold_next_steps']:
                            step_ = [pos_idx if idx==selected_step_con else idx for idx in step_]
                            new_gold_next_steps.append(sorted(step_))

                        new_false_next_steps = []
                        for step_ in combinations(new_pre, 2):
                            step_ = list(sorted(step_))
                            if step_ not in new_gold_next_steps:
                                new_false_next_steps.append(sorted(step_))
                        
                        new_gold_next_steps_abd = []
                        for step_ in now_state['gold_next_steps_abd']:
                            step_ = [pos_idx if idx==selected_step_con else idx for idx in step_]
                            new_gold_next_steps_abd.append(step_)
                                
                        states_positive_true_step.append({
                                'pre':sorted(new_pre),
                                'con':now_state['con'],
                                'depth_to_finish':now_state['depth_to_finish'],
                                'length_to_finish':now_state['length_to_finish'],
                                'state_future_steps':now_state['state_future_steps'],
                                'state_previous_steps':now_state['state_previous_steps'],
                                'state_label':1, # positive state
                                'fact_label':new_fact_label,
                                'fact_dtf':new_fact_dtf,
                                'gold_next_steps':new_gold_next_steps,
                                'false_next_steps':new_false_next_steps,
                                'gold_next_steps_abd':new_gold_next_steps_abd,

                                'modify':[selected_step_con, pos_idx],
                        })
                        
                    if score_ < neg_bleurt_thre:

                        neg_idx = f"module_neg_int{len(add_sents)}"
                        add_sents[neg_idx] = sent_

                        new_pre = []
                        new_fact_label = {}

                        for idx in now_state['pre']:
                            if idx != selected_step_con:
                                new_pre.append(idx)
                                new_fact_label[idx] = 1

                        new_pre.append(neg_idx)
                        new_fact_label[neg_idx] = 0 # negative fact

                        new_fact_dtf = copy.deepcopy(now_state['fact_dtf'])
                        new_fact_dtf[neg_idx] = 100
    
                        states_negative_true_step.append({
                                'pre':sorted(new_pre),
                                'con':now_state['con'],
                                'depth_to_finish':now_state['depth_to_finish'],
                                'length_to_finish':now_state['length_to_finish'],
                                'state_future_steps':now_state['state_future_steps'],
                                'state_previous_steps':now_state['state_previous_steps'],
                                'state_label':0, # negative state
                                'fact_label':new_fact_label,
                                'fact_dtf':new_fact_dtf,
                                'gold_next_steps':now_state['gold_next_steps'],
                                'false_next_steps':now_state['false_next_steps'],
                                'gold_next_steps_abd':now_state['gold_next_steps_abd'],
                            

                                'modify':[selected_step_con, neg_idx],
                        })
        
        data_item['controller_data']['all_sents'].update(add_sents)
        data_item['controller_data']['states_positive_true_step'] = states_positive_true_step
        data_item['controller_data']['states_negative_true_step'] = states_negative_true_step

    with open(buffer_name,'w') as f:
        json.dump(infered_buffer, f, indent=4)


    print("positive state&abduction step -- module --> postive/negative state")
    # compare with gold intermediate sent, bleurt>0.28 and max is positive, bleurt<0.28 and min is negative

    max_state_pre_data_item = 20

    pos_bleurt_thre = 0.28
    neg_bleurt_thre = 0.28

    if reasoning_modules_type == 'bart':
        buffer_name_abd="infered_buffer_abduction.json" # bart
    elif reasoning_modules_type == 't5':
        buffer_name_abd="infered_buffer_abduction_t5.json" #t5


    try:
        infered_buffer_abduction = json.load(open(buffer_name_abd,'r'))
    except:
        infered_buffer_abduction = {} 

    for data_item in tqdm(process_data):
        gt_tree = get_gt_node_list(data_item)
        all_reasoning_steps = [[sorted(node['pre']),node['id']] for node in gt_tree if node['pre']]
        
        add_sents = {}
        states_positive_true_step_abd = []
        states_negative_true_step_abd = []
        
        candidate_states = data_item['controller_data']['states_positive_inter']
        random.shuffle(candidate_states)
        
        for state in candidate_states[:max_state_pre_data_item]:
            now_state = copy.deepcopy(state)
            old_con_id = now_state['con']
            
            find_abd_step = None
            for step in all_reasoning_steps:            
                if len(step[0]) != 2: continue
                if old_con_id in step[0]:
                    find_abd_step = step
            if find_abd_step is None: continue
                
            
            abd_in1 = data_item['controller_data']['all_sents'][find_abd_step[1]]
            abd_in2_id = [p_ for p_ in find_abd_step[0] if p_ != old_con_id][0]
            abd_in2 = data_item['controller_data']['all_sents'][abd_in2_id]
        
            sent1 = abd_in1
            sent2 = abd_in2
            
            buffer_key = sent1+ '+' + sent2
            if buffer_key not in infered_buffer_abduction:
                input_sents = [[add_fullstop(sent1), add_fullstop(sent2)]]
                output_sents, output_types = inference_alltype_batch(input_sents,(reasoning_modules_abd,tokenizer_module_abd,args_module_abd))
                output_sents, output_types = output_sents[0], output_types[0]

                infered_buffer_abduction[buffer_key] = (output_sents, output_types)
            else:
                output_sents, output_types = infered_buffer_abduction[buffer_key]
                    
            gold_con_sents = data_item['controller_data']['all_sents'][old_con_id]
                
            inter_scores = bleurt_scorer.score(references = [gold_con_sents]*len(output_sents), candidates = output_sents)
                
            for sent_, score_ in zip(output_sents, inter_scores):

                if score_ == max(inter_scores) and score_ > pos_bleurt_thre:

                    pos_idx = f"module_pos_abd{len(add_sents)}"
                    add_sents[pos_idx] = sent_

                    new_pre = now_state['pre']
                    new_fact_label = now_state['fact_label']
                    new_fact_dtf = now_state['fact_dtf']
                    new_gold_next_steps = now_state['gold_next_steps']
                    new_false_next_steps = now_state['false_next_steps']

                    new_gold_next_steps_abd = []
                    for step_ in now_state['gold_next_steps_abd']:
                        step_ = [pos_idx if idx==old_con_id else idx for idx in step_]
                        new_gold_next_steps_abd.append(step_)

                    states_positive_true_step_abd.append({
                            'pre':sorted(new_pre),
                            'con':pos_idx,
                            'depth_to_finish':now_state['depth_to_finish'],
                            'length_to_finish':now_state['length_to_finish'],
                            'state_future_steps':now_state['state_future_steps'],
                            'state_previous_steps':now_state['state_previous_steps'],
                            'state_label':1, # positive state
                            'fact_label':new_fact_label,
                            'fact_dtf':new_fact_dtf,
                            'gold_next_steps':new_gold_next_steps,
                            'false_next_steps':new_false_next_steps,
                            'gold_next_steps_abd':new_gold_next_steps_abd,

                            'modify':[old_con_id, pos_idx],
                    })
                    

                if score_ < neg_bleurt_thre:

                    neg_idx = f"module_neg_abd{len(add_sents)}"
                    add_sents[neg_idx] = sent_

                    new_pre = now_state['pre']
                    new_fact_label = {idx:0 for idx in new_pre}
                    new_fact_dtf = {idx:100 for idx in new_pre}

                    states_negative_true_step_abd.append({
                            'pre':sorted(new_pre),
                            'con':neg_idx,
                            'depth_to_finish':now_state['depth_to_finish'],
                            'length_to_finish':now_state['length_to_finish'],
                            'state_future_steps':now_state['state_future_steps'],
                            'state_previous_steps':now_state['state_previous_steps'],
                            'state_label':0, # negative state
                            'fact_label':new_fact_label,
                            'fact_dtf':new_fact_dtf,
                            'gold_next_steps':now_state['gold_next_steps'],
                            'false_next_steps':now_state['false_next_steps'],
                            'gold_next_steps_abd':now_state['gold_next_steps_abd'],

                            'modify':[old_con_id, neg_idx],
                    })
                    
                    
        
        data_item['controller_data']['all_sents'].update(add_sents)
        data_item['controller_data']['states_positive_true_step_abd'] = states_positive_true_step_abd
        data_item['controller_data']['states_negative_true_step_abd'] = states_negative_true_step_abd

    with open(buffer_name_abd,'w') as f:
        json.dump(infered_buffer_abduction, f, indent=4)


    if task_name == 'task_2':
        print("adding distractors for task2")
        for data_item in tqdm(process_data):

            distractors = data_item['meta']['distractors']

            # add distractors_sents
            distractors_sents = {sent_id:add_fullstop(data_item['meta']['triples'][sent_id]) for sent_id in distractors}
            data_item['controller_data']['all_sents'].update(distractors_sents)
            data_item['controller_data']['distractors'] = distractors

            # for each state, add distractors
            for key in ['states_positive_orig','states_positive_inter', 
                        'states_negative_replace', 'states_negative_discard', 
                        'states_positive_true_step', 'states_negative_true_step', 
                        'states_positive_true_step_abd', 'states_negative_true_step_abd', 
                    ]:
                if key not in data_item['controller_data'].keys(): continue
                    
                new_states = []
                for orig_state in data_item['controller_data'][key]:
                    
                    state = copy.deepcopy(orig_state) 
                    
                    state['pre'] = state['pre'] + [sent_id for sent_id in distractors if sent_id not in state['pre']]
                    state['fact_label'].update({sent_id:0 for sent_id in distractors})
                    state['fact_dtf'].update({sent_id:100 for sent_id in distractors})

                    state['gold_next_steps'] = [list(sorted(step_)) for step_ in state['gold_next_steps']]
                    new_false_next_steps = []
                    for step_ in combinations(state['pre'], 2):
                        step_ = list(sorted(step_))
                        if step_ not in state['gold_next_steps']:
                            new_false_next_steps.append(sorted(step_))
                    state['false_next_steps'] = new_false_next_steps

                    state['gold_next_steps_abd'] = state['gold_next_steps_abd']

                    state['con'] = state['con']
                    state['depth_to_finish'] = state['depth_to_finish']
                    state['length_to_finish'] = state['length_to_finish']
                    state['state_future_steps'] = state['state_future_steps']
                    state['state_previous_steps'] = state['state_previous_steps']
                    state['state_label'] = state['state_label']
                    
                    new_states.append(state)
                    
                data_item['controller_data'][key] += new_states 


    with open(save_name,'w') as f:
        for data_item in process_data:
            f.write(json.dumps(data_item)+'\n')
