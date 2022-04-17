import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import argparse
from pprint import pprint
import csv
from datetime import datetime
from collections import defaultdict
import itertools

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer


import bleurt 
from bleurt import score
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

from tree_utils import *
from evaluate_metric import *
from sent_utils import add_fullstop,sent_overlap
from train_Controller import Controller, tokenize_target_fact_sents


# ----- reasoning_module ----- 
from reasoning import load_reasoning_module, module_generate, inference_alltype_batch, inference_alltype_batch_with_buffer


# ----- controller -----
def load_controller(exp_dir,model_name=None):
    # read config
    config = json.load(open(osp.join(exp_dir,'config.json')))
    model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
    args = argparse.Namespace(**config)
    model_config = argparse.Namespace(**model_config)

    # load model
    model = Controller(args,model_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load trained parameters
    if model_name:
        model_path = osp.join(exp_dir, model_name)
    else:
        model_path = osp.join(exp_dir,'best_model.pth')

    state_dict = torch.load(model_path,map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model,tokenizer,args



def contorller_predict_one(target_sents,fact_sents,controller_tuple,fact_score_mask=None):
    controller,tokenizer_controller,args_controller = controller_tuple
    controller.eval()
    
    target_sents = [add_fullstop(s) for s in target_sents]
    fact_sents = [add_fullstop(s) for s in fact_sents]
    
    batch_encoder_inputs = {'input_ids':[],'attention_mask':[]}
    batch_info = []

    input_ids, attention_mask, target_offsets, fact_offsets = \
        tokenize_target_fact_sents(target_sents, fact_sents, tokenizer_controller, pad_max_length=None)

    batch_encoder_inputs['input_ids'].append(input_ids)
    batch_encoder_inputs['attention_mask'].append(attention_mask)
    info_item = {
        'target_offsets':target_offsets,
        'fact_offsets':fact_offsets,
        'fact_score_mask':fact_score_mask if fact_score_mask else [1]*(len(target_sents)*len(fact_sents))
    }
    batch_info.append(info_item)

    batch_encoder_inputs['input_ids'] = torch.LongTensor(batch_encoder_inputs['input_ids']).to(controller.device)
    batch_encoder_inputs['attention_mask'] = torch.LongTensor(batch_encoder_inputs['attention_mask']).to(controller.device)
    
    controller_predict = controller.score_instance(batch_encoder_inputs,batch_info)
    
    batch_pred_state_score = controller_predict['batch_pred_state_score']
    batch_pred_fact_score = controller_predict['batch_pred_fact_score']
    batch_pred_step_score = controller_predict['batch_pred_step_score']
    batch_pred_step_abd_score = controller_predict['batch_pred_step_abd_score']
    
    
    state_score = batch_pred_state_score[0].detach().clone().cpu().data
    fact_score = batch_pred_fact_score[0].detach().clone().cpu().data
    step_score = batch_pred_step_score[0]
    step_score = step_score.detach().clone().cpu().data if step_score is not None else None

    step_abd_score = batch_pred_step_abd_score[0]
    step_abd_score = step_abd_score.detach().clone().cpu().data if step_abd_score is not None else None
    
    return state_score, fact_score, step_score,step_abd_score


def search_task2(data_item,reasoning_module_tuple,reasoning_module_abd_tuple, controller_tuple,\
                 show=False, bleurt_scorer=None,
                 beam_num=1, max_infer_depth=5,abd_depth=0, 
                 num_r=10, buffer_dict=None,
                 step_top_p=0.25, step_top_p_abd=0.1,step_strategy=None,
                 fact_score_thre=0.0, filter_min_fact=2,
                 rerank_strategy=None
                ):
    """

    show: Show the reasoning process
    bleurt_scorer: the bleurt model
    
    beam_num: size of beam
    max_infer_depth: the max reasoning depth

    num_r: the number of the results returned by each module
    buffer_dict: the buffer of the results returned by modules

    step_top_p: the sample rate of deductive steps
    step_top_p_abd: the sample rate of abductive steps

    fact_score_thre: the threshold for filtering distractors
    """
    
    def state2node_tree(state,all_sents):
        """
        convert the state to the entailment tree
        """
        steps = state['state_previous_steps']
        steps_abd = state['state_previous_steps_abd'][::-1]
    
        if len(steps_abd) == 0:
            ### find pred_hypot_id
            candidate_hypot_ids = [step[1] for step in steps if step[1] in state['fact_ids']]
            candidate_hypot_ids_fact_score = [state['fact_score'][state['fact_ids'].index(sent_id)] for sent_id in candidate_hypot_ids]
            max_fact_score_idx = np.argmax(candidate_hypot_ids_fact_score)
            pred_hypot_id = candidate_hypot_ids[max_fact_score_idx]
            
        else:
            ### find final_fact_id
            max_fact_score_idx = np.argmax(state['fact_score'])
            final_fact_id = state['fact_ids'][max_fact_score_idx]
            final_target_id = state['target_ids'][0]

            for (step_abd, pre_abd, step_type) in steps_abd:
                input_sent_id_1 = pre_abd
                input_sent_id_2 = step_abd[1]
                output_sent_id = step_abd[0]
                
                # make the prove continue; adopt the deduction sent; replace the abduction sent with deduction sent
                input_sent_id_1 = input_sent_id_1 if input_sent_id_1 != final_target_id else final_fact_id
                input_sent_id_2 = input_sent_id_2 if input_sent_id_2 != final_target_id else final_fact_id
                output_sent_id = output_sent_id if output_sent_id != final_target_id else final_fact_id
                
                steps.append([(input_sent_id_1,input_sent_id_2),output_sent_id,step_type])
                 
            ### find pred_hypot_id
            pred_hypot_id = steps[-1][1] # predict hypot is the final step's conclusion
        

        pred_node_list = []
        for step_pre, step_con, step_type in steps: # assert the steps are in order
            for idx in step_pre:
                if idx not in [node['id'] for node in pred_node_list]:
                    new_node = {
                        'id':idx,
                        'sent':all_sents[idx],
                        'pre':[],
                        'step_type':[],
                    }
                    pred_node_list.append(new_node)
            idx = step_con
            new_node = {
                'id':idx,
                'sent':all_sents[idx],
                'pre':step_pre,
                'step_type':step_type,
            }
            pred_node_list.append(new_node)

        pred_tree = get_tree(pred_hypot_id,pred_node_list)
            
        return pred_tree

    def add_bleurt_score(state):
        """
        add the bleurt score for between the target H and each fact
        """
        target_sent = state['target_sents'][0]
        fact_sents = state['fact_sents']
        state['bleurt'] = bleurt_scorer.score(references = [target_sent]*len(fact_sents), candidates=fact_sents)
        return state

    def remove_distractor_by_fact_score(state, thre = 0.01, filter_min_fact=2):
        """
        remove the facts whose scores are lower then the threshold
        """
        new_state = {}

        new_state['target_sents'] = state['target_sents']
        new_state['target_ids'] = state['target_ids']
        new_state['state_previous_steps'] = state['state_previous_steps']
        new_state['state_previous_steps_abd'] = state['state_previous_steps_abd']
        
        new_fact_id = []
        new_fact_sent = []
        for fact_id, fact_sent, fact_score in zip(state['fact_ids'],state['fact_sents'],state['fact_score']):
            if fact_score > thre:
                new_fact_id.append(fact_id)
                new_fact_sent.append(fact_sent)
                
        if len(new_fact_sent) > 1:
            new_state['fact_ids'] = new_fact_id
            new_state['fact_sents'] = new_fact_sent
        else:
            new_fact_id = []
            new_fact_sent = []
            sorted_index = list(np.argsort(state['fact_score']))[::-1] # 得分从大到小
            for idx in sorted_index[:filter_min_fact]: 
                new_fact_id.append(state['fact_ids'][idx])
                new_fact_sent.append(state['fact_sents'][idx])
                
            new_state['fact_ids'] = new_fact_id
            new_state['fact_sents'] = new_fact_sent
        
        return new_state
    
    def check_state_too_long(state):
        input_str = " ".join(state['target_sents'] + state['fact_sents'])
        word_num = len(input_str.split())
        if word_num > 400:
            print('INPUT TOO LONG')
            new_state = copy.deepcopy(state)
            new_state['fact_sents'] = new_state['fact_sents'][:-5]
            new_state['fact_ids'] = new_state['fact_ids'][:-5]
            return new_state
        else:
            return state
        
    def post_process_module_output(o_sent):
        if '.' in o_sent:
            o_sent = o_sent.split('.')[0] + '.'
        return o_sent
        
        
        
    (controller,tokenizer_controller,args_controller) = controller_tuple
    (reasoning_modules,tokenizer_module,args_module) = reasoning_module_tuple
    
    H = add_fullstop(data_item['hypothesis'])
    QA = data_item['question'] + ' ' + data_item['answer']
    Sents = get_sent_list(data_item)
    
    beam_states = []
    all_sents = {}
    all_sents.update(Sents)
    all_sents['H'] = H
    all_sents = {k:add_fullstop(v) for k,v in all_sents.items()}

    ##### ----- init: make initial state -----
    init_state = {
        'target_sents':[H],
        'target_ids':['H'],
        'fact_sents':[add_fullstop(v) for k,v in Sents.items()],
        'fact_ids':[k for k,v in Sents.items()],
        'state_previous_steps':[],
        'state_previous_steps_abd':[],
    }
    init_state = check_state_too_long(init_state) # if too long, remove last 5 facts

    state_score, fact_score, step_score,step_abd_score = contorller_predict_one(init_state['target_sents'], init_state['fact_sents'], 
                                                                                 controller_tuple)
    init_state['state_score'] = state_score
    init_state['fact_score'] = fact_score
    init_state['step_score'] = step_score
    init_state['step_abd_score'] = step_abd_score

    ### for init state, remove_distractor_by_fact_score
    if fact_score_thre > 0.0:
        new_init_state = remove_distractor_by_fact_score(init_state, fact_score_thre,filter_min_fact)
        state_score, fact_score, step_score,step_abd_score = contorller_predict_one(new_init_state['target_sents'], new_init_state['fact_sents'], 
                                                                                     controller_tuple)
        new_init_state['state_score'] = state_score
        new_init_state['fact_score'] = fact_score
        new_init_state['step_score'] = step_score
        new_init_state['step_abd_score'] = step_abd_score
        init_state = new_init_state
    ###

    if bleurt_scorer:
        init_state = add_bleurt_score(init_state)

    init_fact_length = len(init_state['fact_ids'])
    beam_states.append([init_state])

    ### only one facts 
    if len(init_state['fact_ids']) == 1:
        sent_id = init_state['fact_ids'][0]
        new_sent_id = 'con-'+sent_id
        pred_tree = [
            {
                'id':new_sent_id,
                'sent':Sents[sent_id],
                'pre':[(sent_id)]
            },
            {
                'id':sent_id,
                'sent':Sents[sent_id],
                'pre':[]
            }]
        all_pred_tree = [pred_tree]
        return pred_tree, all_pred_tree
    ### 

    if show:
        pprint(init_state,width=200)

    # search untill the max_infer_depth or all facts are used
    for beam_search_step in range(min(max_infer_depth, init_fact_length-1)):

        all_sents = {k:add_fullstop(v) for k,v in all_sents.items()}

        candidate_states = []
        for state in beam_states[-1]:


            ###  comapre all deduction step and abduction step and compute a threshold socre for top steps
            step_thre = 0
            step_thre_abd = 0

            all_step_scores = []
            if state['step_score'] is not None: all_step_scores += state['step_score']
            if state['step_abd_score'] is not None: all_step_scores += state['step_abd_score']
            all_step_scores = sorted(all_step_scores,reverse=True)

            # if step_strategy == 2:
            step_thre = 0
            step_thre_abd = max(state['step_score'])    
            ###


            ##### ----- Deduction step -----
            # for each state, select steps
            state_all_steps = list(itertools.combinations(state['fact_ids'],2)) # the same order with Controller

            if state['step_score'] is not None:
                assert len(state['step_score']) == len(state_all_steps)
                sorted_index = np.argsort(state['step_score']) 
                sorted_all_steps = []
                for idx in list(sorted_index)[::-1]: # from large score to small score
                    step_ = state_all_steps[idx]
                    score_ = state['step_score'][idx]
                    if score_ >= step_thre: # select step with score larger then threshold
                        sorted_all_steps.append(step_)
                state_all_steps = sorted_all_steps

            input_sents = []
            selected_steps = []
            for step in state_all_steps:
                sent1 = all_sents[step[0]]
                sent2 = all_sents[step[1]]
                if sent_overlap(sent1,sent2,spacy_nlp,thre=0.75):
                    input_sents.append([sent1,sent2])
                    selected_steps.append(step)
                else: 
                    pass 
                    # print(f"filter out step: {sent1} & {sent2}")

            top_k = 1+int(step_top_p*len(selected_steps)) # set top_k by top_p
            selected_steps = selected_steps[:top_k]
            input_sents = input_sents[:top_k]

            if show:
                print("select deduction step:",selected_steps)

            if len(selected_steps) == 0:
                # if no step meet the requirement, select the top-1 step with no requirement
                # sorted_index = np.argsort(state['step_score']) 
                # idx = sorted_index[0]
                # selected_steps = [state_all_steps[idx]]
                selected_steps = state_all_steps
                input_sents = [
                    [all_sents[step[0]],all_sents[step[1]]] \
                    for step in selected_steps
                ]


            ## reasoning with modules (with buffer dict to speed up)
            output_sents, output_type, buffer_dict = inference_alltype_batch_with_buffer(input_sents,reasoning_module_tuple, \
                                                                                         num_return=num_r, buffer_dict=buffer_dict)

            # make candidate states
            for i_sents, step, o_sents, o_types in zip(input_sents, selected_steps, output_sents, output_type):
                scores = []
                for o_sent, o_type in zip(o_sents, o_types):  
                    o_sent = post_process_module_output(o_sent)
                    
                    new_id = f"pred_int{len(all_sents)}"
                    all_sents[new_id] = o_sent

                    # make new fact set
                    save_ids = [idx for idx in state['fact_ids'] if idx not in step]
                    new_fact_ids = save_ids + [new_id]
                    new_fact_sents = [all_sents[idx] for idx in new_fact_ids]

                    new_state = {
                        'target_sents':state['target_sents'],
                        'target_ids':state['target_ids'],
                        'fact_sents':new_fact_sents,
                        'fact_ids':new_fact_ids,
                        'state_previous_steps':state['state_previous_steps'] + [[step,new_id,o_type]],
                        'state_previous_steps_abd':state['state_previous_steps_abd']
                    }

                    # make controller prediction
                    state_score, fact_score, step_score, step_abd_score = contorller_predict_one(new_state['target_sents'],new_state['fact_sents'],
                                                                                 controller_tuple)
                    new_state['state_score'] = state_score
                    new_state['fact_score'] = fact_score
                    new_state['step_score'] = step_score
                    new_state['step_abd_score'] = step_abd_score


                    if o_sent.count('.') > 1:
                        print("skip by .")
                        continue # for o_sent, o_type in zip(o_sents, o_types)

                    candidate_states.append(new_state)

                    
                    
            ##### ----- Abduction step ----- 
            if len(state['state_previous_steps_abd']) < abd_depth and len(state['fact_ids']) > 2:

                state_all_steps_abd = [[state['target_ids'][0], p_] for p_ in state['fact_ids']]  # the same order with Controller

                if state['step_abd_score'] is not None:
                    assert len(state['step_abd_score']) == len(state_all_steps_abd)
                    sorted_index = np.argsort(state['step_abd_score'])
                    sorted_all_steps_abd = []
                    for idx in list(sorted_index)[::-1]: # from large score to small score
                        step_abd_ = state_all_steps_abd[idx]
                        score_abd_ = state['step_abd_score'][idx]
                        if score_abd_ >= step_thre_abd: # select abduction step with score larger then threshold
                            sorted_all_steps_abd.append(step_abd_)
                    state_all_steps_abd = sorted_all_steps_abd


                input_sents = []
                selected_steps = []
                for step in state_all_steps_abd:
                    sent1 = all_sents[step[0]] # sent1 is the conclusion
                    sent2 = all_sents[step[1]]
                    if sent_overlap(sent1,sent2,spacy_nlp,thre=0.75):
                        input_sents.append([sent1,sent2])
                        selected_steps.append(step)
                    else: 
                        pass 

                top_k = 1+int(step_top_p_abd*len(selected_steps))
                selected_steps = selected_steps[:top_k]
                input_sents = input_sents[:top_k]

                if len(selected_steps) == 0:
                    pass # if no step meet the requirement for abduction, just continue

                if show:
                    print("select abduction step:",selected_steps)

                # reasoning with abduction modules (with buffer dict to speed up)
                output_sents, output_type, buffer_dict = inference_alltype_batch_with_buffer(input_sents,reasoning_module_abd_tuple, \
                                                                                num_return=num_r, buffer_dict=buffer_dict)

                # make candidate states for abduction
                for i_sents, step_abd, o_sents, o_types in zip(input_sents, selected_steps, output_sents, output_type):
                    for o_sent, o_type in zip(o_sents, o_types):
                        o_sent = post_process_module_output(o_sent)

                        new_id = f"pred_abd{len(all_sents)}"
                        all_sents[new_id] = o_sent

                        # make new fact set
                        abd_pre_sent_id = step_abd[1] # step_abd e.g., ['H',sent1]
                        save_ids = [idx for idx in state['fact_ids'] if idx != abd_pre_sent_id]
                        new_fact_ids = save_ids
                        new_fact_sents = [all_sents[idx] for idx in new_fact_ids]

                        # make new target set
                        new_target_sents = [o_sent]
                        new_target_ids = [new_id]

                        new_state = {
                            'target_sents':new_target_sents,
                            'target_ids':new_target_ids,
                            'fact_sents':new_fact_sents,
                            'fact_ids':new_fact_ids,
                            'state_previous_steps':state['state_previous_steps'],
                            'state_previous_steps_abd':state['state_previous_steps_abd'] + [[step_abd, new_id,o_type]],
                        }

                        # make controller prediction
                        state_score, fact_score, step_score, step_abd_score = contorller_predict_one(new_state['target_sents'],new_state['fact_sents'],
                                                                                     controller_tuple,fact_score_mask=None)
                        new_state['state_score'] = state_score
                        new_state['fact_score'] = fact_score
                        new_state['step_score'] = step_score
                        new_state['step_abd_score'] = step_abd_score

                        
                        if o_sent.count('.') > 1:
                            print("skip by . (abd)")
                            continue # for o_sent, o_type in zip(o_sents, o_types)

                        candidate_states.append(new_state)


        if len(candidate_states) == 0:
            candidate_states.append(new_state)

        # select top BEAM_NUM state by state score
        candidate_states = sorted(candidate_states,key=lambda x:x['state_score'],reverse=True)
        selected_new_states = candidate_states[:beam_num]

        if bleurt_scorer:
            selected_new_states = [add_bleurt_score(state) for state in selected_new_states]

        beam_states.append(selected_new_states)

        if show:
            print(beam_search_step,'='*50)
            for state in candidate_states:
                print('-'*10)                
                print(float(state['state_score']))
                print('deduction steps',state['state_previous_steps'])
                print('abduction steps',state['state_previous_steps_abd'])
                print(state['target_ids'])
                print(state['fact_ids'],state['fact_score'])


    # for task2, select finnal state from all beams
    all_beam_states = []
    for states in beam_states:
        all_beam_states += [state for state in states if len(state['state_previous_steps'])>0]

    if rerank_strategy == 'bleurt_long':
        final_states = sorted(all_beam_states,key=lambda x:(max(x['bleurt'])+(+0.01*len(x['state_previous_steps']+x['state_previous_steps_abd']))),reverse=True)
    else:
        raise NotImplemented

    if show:
        print('='*100)
        for state in final_states:
            print(max(state['fact_score']),state['state_previous_steps'],state['state_previous_steps_abd'])

        print('='*100)

    # convert state to tree
    state = final_states[0]
    pred_tree = state2node_tree(state, all_sents)

    print('abduction steps',state['state_previous_steps_abd'])

    if show:
        print(state)
        print("Final",'='*50)
        print(float(state['state_score']))
        print_node_tree(pred_tree[0]['id'],pred_tree)

    all_pred_tree = []
    for state in final_states:
        all_pred_tree.append(state2node_tree(state, all_sents))

    return pred_tree, all_pred_tree

def run(args):

    print("loading BLEURT")
    bleurt_scorer = score.BleurtScorer(args.bleurt_path)

    device = 'cuda'


    if args.module_types == 'separate_all':
        print("loading reasoning modules")
        module_sub,_,_ = load_reasoning_module(args.exp_dir_sub)
        module_conj,_,_ = load_reasoning_module(args.exp_dir_conj)
        module_if,tokenizer_module,args_module = load_reasoning_module(args.exp_dir_if)

        reasoning_modules = {
            'substitution':{
                'model':module_sub.to(device),
                'task_prefix':'',
            },
            'conjunction': {
                'model':module_conj.to(device),
                'task_prefix':'',
            },
            'if-then':{
                'model':module_if.to(device),
                'task_prefix':'',
            },
        }

        print("loading abduction reasoning modules")
        module_sub_abd,_,_ = load_reasoning_module(args.exp_dir_sub_abd)
        module_conj_abd,_,_ = load_reasoning_module(args.exp_dir_conj_abd)
        module_if_abd,tokenizer_module_abd,args_module_abd = load_reasoning_module(args.exp_dir_if_abd)

        reasoning_modules_abd = {
            'substitution_abd':{
                'model':module_sub_abd.to(device),
                'task_prefix':'',
            },
            'conjunction_abd': {
                'model':module_conj_abd.to(device),
                'task_prefix':'',
            },
            'if-then_abd':{
                'model':module_if_abd.to(device),
                'task_prefix':'',
            },
        }
    elif args.module_types == 'separate':
        print("loading deductive reasoning modules")
        module_ded,tokenizer_module,args_module = load_reasoning_module(args.exp_dir_ded)
        reasoning_modules = {
            'substitution':{
                'model':module_ded.to(device),
                'task_prefix':'deductive substitution:',
            },
            'conjunction': {
                'model':module_ded.to(device),
                'task_prefix':'deductive conjunction:',
            },
            'if-then':{
                'model':module_ded.to(device),
                'task_prefix':'deductive if-then:',
            },
        }

        # abduction module
        print("loading abductive reasoning modules")
        module_abd,tokenizer_module_abd,args_module_abd = load_reasoning_module(args.exp_dir_abd)
        reasoning_modules_abd = {
            'substitution_abd':{
                'model':module_abd.to(device),
                'task_prefix':'abductive substitution:',
            },
            'conjunction_abd': {
                'model':module_abd.to(device),
                'task_prefix':'abductive conjunction:',
            },
            'if-then_abd':{
                'model':module_abd.to(device),
                'task_prefix':'abductive if-then:',
            },
        }

    elif args.module_types == 'separate_notype':
        print("loading deductive reasoning modules")
        module_ded,tokenizer_module,args_module = load_reasoning_module(args.exp_dir_ded)
        reasoning_modules = {
            'notype':{
                'model':module_ded.to(device),
                'task_prefix':'',
            },
        }
        # abduction module
        print("loading abductive reasoning modules")
        module_abd,tokenizer_module_abd,args_module_abd = load_reasoning_module(args.exp_dir_abd)
        reasoning_modules_abd = {
            'notype_abd':{
                'model':module_abd.to(device),
                'task_prefix':'',
            },
        }

    elif args.module_types == 'single':
        print("loading single module for deductive and abductive reasoning")
        module_all,tokenizer_module,args_module = load_reasoning_module(args.exp_dir_single)
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

    elif args.module_types == 'single_notype':
        print("loading single module for deductive and abductive reasoning with notype")
        module_all,tokenizer_module,args_module = load_reasoning_module(args.exp_dir_single)
        reasoning_modules = {
            'single_notype':{
                'model':module_all.to(device),
                'task_prefix':'deductive:',
            },
        }

        # abduction module
        tokenizer_module_abd,args_module_abd = tokenizer_module,args_module
        reasoning_modules_abd = {
            'single_notype_abd':{
                'model':module_all.to(device),
                'task_prefix':'abductive:',
            },
        }
        
    else:
        raise NotImplemented


    print("loading controller")
    controller,tokenizer_controller,args_controller = load_controller(args.exp_dir_controller, args.model_name_controller)
    
    controller = controller.to(device)
    controller.device = controller.encoder.device

    print("loading data")
    with open(args.data_file,'r') as f:
        datas = [json.loads(line) for line in f.readlines()]
        
    controller_tuple = (controller,tokenizer_controller,args_controller)
    reasoning_module_tuple = (reasoning_modules,tokenizer_module,args_module)
    reasoning_module_abd_tuple = (reasoning_modules_abd,tokenizer_module_abd,args_module_abd)
    
    print("loading reasoning buffer")
    if not os.path.exists(args.buffer_file):
        print("init buffer file")
        json.dump({},open(args.buffer_file,'w'))
    buffer_dict = json.load(open(args.buffer_file,'r'))

    pred_trees = []
    gold_trees = []
    eval_results = []
    eval_datas = datas

    all_pred_trees = []

    for i, data_item in enumerate(eval_datas):
        gold_tree = get_gt_node_list(data_item)
        pred_tree, all_pred_tree = search_task2(data_item,reasoning_module_tuple,reasoning_module_abd_tuple,controller_tuple,\
                                    beam_num=args.beam_num,step_top_p=args.step_top_p,step_top_p_abd=args.step_top_p_abd,\
                                        num_r=args.num_r,abd_depth=args.abd_depth,buffer_dict=buffer_dict,\
                                        step_strategy=args.step_strategy,\
                                        fact_score_thre=args.fact_score_thre,\
                                        bleurt_scorer=bleurt_scorer,\
                                        rerank_strategy=args.rerank_strategy,\
                                        filter_min_fact=args.filter_min_fact, \
                                        max_infer_depth=args.max_infer_depth)

        eval_result = eval_tree(pred_tree,gold_tree,bleurt_scorer)
        
        gold_trees.append(gold_tree)
        pred_trees.append(pred_tree)
        eval_results.append(eval_result)

        all_pred_trees.append(all_pred_tree)
        
        print(f'{len(eval_results)} / {len(eval_datas)}')
        print('\ngold_tree:')
        print_node_tree(gold_tree[0]['id'],gold_tree)
        print('\npred_tree:')
        print_node_tree(pred_tree[0]['id'],pred_tree)
        print()
        print(collect_results(eval_results))
        print()

        if not i%50:
            print("save buffer")
            buffer_dict_reread = json.load(open(args.buffer_file,'r'))
            buffer_dict_reread.update(buffer_dict)
            json.dump(buffer_dict_reread,open(args.buffer_file,'w'),indent=4)

    final_result = collect_results(eval_results)

    # save buffer dict
    buffer_dict_reread = json.load(open(args.buffer_file,'r'))
    buffer_dict_reread.update(buffer_dict)
    json.dump(buffer_dict_reread,open(args.buffer_file,'w'),indent=4)
    
    # add args info
    final_result['args'] = vars(args)

    # group by steps
    results_by_steps = defaultdict(list)
    for data_item, eval_result in zip(eval_datas,eval_results):
        length_of_proof = len(parse_proof(data_item['meta']['step_proof']))
        results_by_steps[length_of_proof].append(eval_result)
    results_by_steps = dict(results_by_steps)
    for k,v in results_by_steps.items():
        results_by_steps[k] = collect_results(v)
    final_result['results_by_steps'] = results_by_steps

    # save result
    timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = osp.join(args.exp_dir_controller, args.save_dir_name)
    os.makedirs(save_dir,exist_ok=True)
    save_result_path = osp.join(save_dir, f'{args.model_name_controller}.{timestr}.json')
    with open(save_result_path,'w') as f:
        json.dump(final_result,f,indent=4)

    # make detail results
    if args.save_details:
        # save csv results
        csv_results = []
        for pt,data_item in zip(pred_trees, eval_datas):
            r = convert_to_result(pt,data_item)
            csv_results.append(r)
                
        results_tsv = ["$proof$ = " + item['proof'] for item in csv_results]
        with open(save_result_path.replace('json','csv'),'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            for r in results_tsv:
                tsv_w.writerow([r])  

        # save pred trees and results
        with open(save_result_path.replace('json','details.json'),'w') as f:
            for gt,pt,r in zip(gold_trees,pred_trees,eval_results):
                f.write(json.dumps({
                    'gold_tree':gt,
                    'pred_tree':pt,
                    'score':r,
                })+'\n')





def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Reasoning')

    # dateset
    parser.add_argument("--data_file", type=str, default='')   
    
    # model
    parser.add_argument("--module_types", type=str, default="separate_all") 

    parser.add_argument("--exp_dir_sub", type=str, default="")  
    parser.add_argument("--exp_dir_conj", type=str, default="")  
    parser.add_argument("--exp_dir_if", type=str, default="")  
    parser.add_argument("--exp_dir_sub_abd", type=str, default="")  
    parser.add_argument("--exp_dir_conj_abd", type=str, default="")  
    parser.add_argument("--exp_dir_if_abd", type=str, default="")  

    parser.add_argument("--exp_dir_ded", type=str, default="")  
    parser.add_argument("--exp_dir_abd", type=str, default="")  

    parser.add_argument("--exp_dir_single", type=str, default="") 

    parser.add_argument("--buffer_file", type=str, default="") 

    parser.add_argument("--exp_dir_controller", type=str, default="") 
    parser.add_argument("--model_name_controller", type=str, default=None) 
    
    parser.add_argument("--bleurt_path", type=str, default="../../bleurt/bleurt-large-512")  

    parser.add_argument("--beam_num", type=int, default=1) 
    parser.add_argument("--step_top_p", type=float, default=0.1) 
    parser.add_argument("--step_top_p_abd", type=float, default=0.1) 
    parser.add_argument("--fact_score_thre", type=float, default=0.0) 
    parser.add_argument("--max_infer_depth", type=int, default=5) 
    

    # set default
    parser.add_argument("--step_strategy", type=int, default=2) 
    parser.add_argument("--rerank_strategy", type=str, default="bleurt_long") 
    parser.add_argument("--filter_min_fact", type=int, default=2) 
    parser.add_argument("--abd_depth", type=int, default=1) 
    parser.add_argument("--num_r", type=int, default=10) 

    parser.add_argument("--save_dir_name", type=str, default="eval")  
    parser.add_argument("--save_details", action='store_true', default=False)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_params()

    if args.model_name_controller == "":
        args.model_name_controller = 'best_model.eval.json'

    run(args)
