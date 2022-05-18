import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import os.path as osp
import json
import argparse

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
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

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

from tree_utils import *
from evaluate_metric import *
from sent_utils import add_fullstop,sent_overlap



# ----- reasoning_module ----- 
def load_reasoning_module(exp_dir):
    # read config
    config = json.load(open(osp.join(exp_dir,'config.json')))
    model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
    args = argparse.Namespace(**config)

    # load model
    if args.model_name_or_path in ['facebook/bart-large']:
        try:
            model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        except:
            model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path in ['t5-large','t5-base','t5-small']:
        try:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        except:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.sep_token = tokenizer.eos_token
    else:
        raise NotImplementedError

    model.config.update(model_config)

    # load trained parameters
    state_dict = torch.load(osp.join(exp_dir,'best_model.pth'),map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model,tokenizer,args

def module_generate(input_sents,model,tokenizer,args,num_return=1):
    model.eval()
    
    with torch.no_grad():
        if args.input_join:
            input_sents = [' '.join(sents) for sents in input_sents]   
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
            num_return_sequences = num_return, 
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    return decoded

def inference_alltype_batch(input_sents,reasoning_module_tuple, bs = 1, num_return=1):
    
    reasoning_modules,tokenizer_module,args_module = reasoning_module_tuple
    # input_sents [num_sent]
    # output_sents [num_sent, num_outputs]
    # cls_types [num_sent, num_outputs]

    # use corresponding module to infer result
    output_sents = [[] for _ in range(len(input_sents))]
    cls_types = [[] for _ in range(len(input_sents))]
    
    for module_type, infer_module in reasoning_modules.items():
        tmp_sents = input_sents
        tmp_outputs = []
        for batch in chunk(tmp_sents,bs):
            generated_sents = module_generate(batch,model = infer_module,tokenizer=tokenizer_module,args=args_module,num_return=num_return)
            # generated_sents len = num_return*bs
            for i in range(0,len(generated_sents),num_return):
                tmp_outputs.append(list(set(generated_sents[i:i+num_return])))
        for index, outs in enumerate(tmp_outputs):
            for out in outs:
                output_sents[index].append(out)
                cls_types[index].append(module_type)
    
    return output_sents,cls_types

def inference_alltype_batch_with_buffer(input_sents,reasoning_module_tuple, num_return=1, buffer_dict=None):
    # input_sents [num_sent]
    # output_sents [num_sent, num_outputs]
    # cls_types [num_sent, num_outputs]

    reasoning_modules,tokenizer_module,args_module = reasoning_module_tuple

    if buffer_dict is None:
        buffer_dict = {}

    for module_type in reasoning_modules.keys():
        if module_type not in buffer_dict:
            buffer_dict[module_type] = {}

    output_sents = [[] for _ in range(len(input_sents))]
    cls_types = [[] for _ in range(len(input_sents))]
    
    # use corresponding module to infer result
    for module_type, infer_module_info in reasoning_modules.items():
        infer_module = infer_module_info['model']
        task_prefix = infer_module_info['task_prefix']
        
        if len(task_prefix) == 0:
            tmp_sents = input_sents
        else:
            tmp_sents = [[task_prefix]+input_ for input_ in input_sents]

        # generate num_return sents for this module_type for each input
        tmp_outputs = [[] for _ in range(len(input_sents))]
        for index, input_ in enumerate(tmp_sents):
            buffer_key = "+".join(input_)
            if buffer_key in buffer_dict[module_type].keys():
                tmp_outputs[index] = buffer_dict[module_type][buffer_key]
            else:
                batch = [input_]
                generated_sents = module_generate(batch,model = infer_module,tokenizer=tokenizer_module,args=args_module,num_return=num_return)
                # generated_sents len = num_return*1
                tmp_outputs[index] = list(set(generated_sents[0:0+num_return]))

                buffer_dict[module_type][buffer_key] = tmp_outputs[index]


        for index, outs in enumerate(tmp_outputs):
            for out in outs:
                output_sents[index].append(out)
                cls_types[index].append(module_type)
    
    return output_sents, cls_types, buffer_dict