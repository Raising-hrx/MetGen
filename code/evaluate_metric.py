import json
from pprint import pprint
from collections import defaultdict
import random
import numpy as np
import os
import copy
from tree_utils import *


def eval_tree(pred_tree,gold_tree,bleurt_scorer,bleurt_process_fn=None,bleurt_threshold=0.28):
    eval_result = {}

    # -----alignment algorithm-----
    pred_int_to_gold_int_mapping = {}
    prediction_to_aligned_gold_sents = {}
    prediction_to_perfect_match = {}

    # the first node is root node which corresponding to the hypothesis
    # modify the pred_tree hypothesis sents
    pred_root_id = pred_tree[0]['id']
    gold_root_id = gold_tree[0]['id']
    gold_hypothesis_sent = gold_tree[0]['sent']
    pred_tree[0]['orig_sent'] = copy.deepcopy(pred_tree[0]['sent'])
    pred_tree[0]['sent'] = gold_hypothesis_sent

    # collect intermediates node set 
    # include the hypot(root) node in the intermediate node set !!!
    pred_inters = [node for node in pred_tree if not node['id'].startswith('sent')]
    gold_inters = [node for node in gold_tree if not node['id'].startswith('sent')]

    for pred_int in pred_inters[::-1]:  # from int* to hypothesis
        # find leaf with max Jaccard similarity
        max_Jaccard = -1
        max_id = None
        pred_leaves_ids = get_leaves_ids(pred_int['id'], pred_tree)
        for gold_int in gold_inters:
            gold_leaves_ids = get_leaves_ids(gold_int['id'], gold_tree)
            j = Jaccard(pred_leaves_ids, gold_leaves_ids)
            if j > max_Jaccard:  # > or >= is different
                max_Jaccard = j
                max_id = gold_int['id']

        if max_Jaccard > 0:
    #         print(pred_int['id'],max_id,max_Jaccard)
            pred_int_to_gold_int_mapping[pred_int['id']] = max_id
            prediction_to_aligned_gold_sents[pred_int['sent']] = get_node(max_id,gold_tree)['sent']
        else:
            # if max Jaccard is 0, align to a dummy gold node with a blank conclusion
            pred_int_to_gold_int_mapping[pred_int['id']] = "NO_MATCH"
            prediction_to_aligned_gold_sents[pred_int['sent']] = ""

        prediction_to_perfect_match[pred_int['sent']] = True if max_Jaccard == 1.0 else False


    # -----Leaves Evaluation-----
    # when multiple pred hit the same gold, one of them is TP, and the others are FP
    gold_leaves = set(get_leaves_ids(gold_tree[0]['id'], gold_tree))
    pred_leaves = set(get_leaves_ids(pred_tree[0]['id'], pred_tree))

    leaves_match = len([leaf for leaf in gold_leaves if leaf in pred_leaves]) # len(pred_leaves.intersection(gold_leaves))
    leaves_p, leaves_r, leaves_f1 = compute_f1(leaves_match, len(pred_leaves), len(gold_leaves))
    leaves_ac = 0 if leaves_f1 < 1.0 else 1
    leaves_compare = {'pred':list(pred_leaves),'gold':list(gold_leaves)}

    # -----Steps Evaluation-----
    # when multiple pred hit the same gold, one of them is TP, and the others are FP
    # include the hypot(root) node
    gold_inters = [node for node in gold_tree if not node['id'].startswith('sent')]
    pred_inters = [node for node in pred_tree if not node['id'].startswith('sent')]

    gold_steps = [[sorted(node['pre']),node['id']] for node in gold_inters]
    pred_steps = [[sorted(node['pre']),node['id']] for node in pred_inters]

    aligned_pred_steps = []
    for step_inputs, step_output in pred_steps:
        aligned_inputs = []
        for idx in step_inputs:
            if idx.startswith('sent'):
                aligned_inputs.append(idx)
            else:
                aligned_inputs.append(pred_int_to_gold_int_mapping[idx])
        aligned_pred_steps.append([sorted(aligned_inputs),pred_int_to_gold_int_mapping[step_output]])

    steps_match = len([step for step in gold_steps if step in aligned_pred_steps])
    steps_p, steps_r, steps_f1 = compute_f1(steps_match, len(pred_steps), len(gold_steps))
    steps_ac = 0 if steps_f1 < 1.0 else 1
    stpes_compare = {'pred':aligned_pred_steps,'gold':gold_steps}

    # -----Intermediates Evaluation-----
    # compute TP for pred and gold, respectively
    pred_sents = []
    gold_sents = []

    pred_hit_sents = set()
    gold_hit_sents = set()

    bleurt_scores_collecter = []
    bleurt_scores_perfect_align_collecter = []
    num_bleurt_correct = 0

    for pred, gold in prediction_to_aligned_gold_sents.items():
        if bleurt_process_fn:
            pred_s = bleurt_process_fn(pred)
            gold_s = bleurt_process_fn(gold)
        else:
            pred_s, gold_s = pred, gold
            
        pred_sents.append(pred_s)
        gold_sents.append(gold_s)

        score = bleurt_scorer.score(references = [gold_s], candidates=[pred_s])[0]
        bleurt_scores_collecter.append(score)
        if prediction_to_perfect_match[pred]:
            bleurt_scores_perfect_align_collecter.append(score)

        if score > bleurt_threshold:
            num_bleurt_correct+=1
            if gold != '':
                pred_hit_sents.add(pred_s)
                gold_hit_sents.add(gold_s)

    inter_p = len(pred_hit_sents)/max(1, len(pred_inters))
    inter_r = len(gold_hit_sents)/max(1, len(gold_inters))
    inter_f1 = div(2 * inter_p * inter_r, inter_p + inter_r)
    inter_ac = 0 if inter_f1 < 1.0 else 1
    # inter_ac = int(num_bleurt_correct==len(prediction_to_aligned_gold_sents.keys()))  # bug for offical evaluation !!!

    inter_mean_bleurt = np.sum(bleurt_scores_collecter) / max(1,len(bleurt_scores_collecter))
    inter_mean_bleurt_perfect_align = np.sum(bleurt_scores_perfect_align_collecter) / max(1,len(bleurt_scores_perfect_align_collecter))
    fraction_perfect_align = np.sum(list(prediction_to_perfect_match.values())) / max(1,len(prediction_to_aligned_gold_sents))

    # -----Overall Evaluation-----
    overall_ac = leaves_ac * steps_ac * inter_ac
    
    
    # -----collect result-----
    eval_result['leaves_p'] = leaves_p
    eval_result['leaves_r'] = leaves_r
    eval_result['leaves_f1'] = leaves_f1
    eval_result['leaves_ac'] = leaves_ac
    eval_result['leaves_compare'] = leaves_compare

    eval_result['steps_p'] = steps_p
    eval_result['steps_r'] = steps_r
    eval_result['steps_f1'] = steps_f1
    eval_result['steps_ac'] = steps_ac
    eval_result['stpes_compare'] = stpes_compare

    eval_result['inter_p'] = inter_p
    eval_result['inter_r'] = inter_r
    eval_result['inter_f1'] = inter_f1
    eval_result['inter_ac'] = inter_ac
    eval_result['inter_mean_bleurt'] = inter_mean_bleurt
    eval_result['inter_mean_bleurt_perfect_align'] = inter_mean_bleurt_perfect_align
    eval_result['fraction_perfect_align'] = fraction_perfect_align
    
    eval_result['overall_ac'] = overall_ac

    eval_result['pred_int_to_gold_int_mapping'] = pred_int_to_gold_int_mapping
    eval_result['prediction_to_aligned_gold_sents'] = prediction_to_aligned_gold_sents
    eval_result['prediction_to_perfect_match'] = prediction_to_perfect_match
    
    return eval_result


def collect_results(eval_results):
    average_keys = ['leaves_p', 'leaves_r', 'leaves_f1', 'leaves_ac', \
                    'steps_p', 'steps_r', 'steps_f1', 'steps_ac', \
                    'inter_p', 'inter_r', 'inter_f1', 'inter_ac', 'inter_mean_bleurt', 'inter_mean_bleurt_perfect_align', \
                    'overall_ac',
                    'fraction_perfect_align']
    average_result = {} 
    eval_results = copy.deepcopy(eval_results)
    for k in average_keys:
        average_result[k] = np.sum([r[k] for r in eval_results]) / len(eval_results)
        average_result[k] = round(average_result[k], 4)
        
    return average_result

def preprocess_pred_tree_task3(pred_tree, gold_tree):
    pred_tree = rename_node(pred_tree)
    
    gold_leaf_sent2id = {}
    for node in gold_tree:
        if node['id'].startswith('sent'):
            gold_leaf_sent2id[node['sent']] = node['id']

    id_map = {}
    for node in pred_tree:
        if len(node['pre']) != 0: continue
        if node['sent'] in gold_leaf_sent2id.keys():
            id_map[node['id']] = gold_leaf_sent2id[node['sent']]
            
    pred_tree = rename_node(pred_tree, id_map)

    # set hypothesis sents
    pred_tree[0]['sent'] = gold_tree[0]['sent']
    
    return pred_tree