#coding:utf8
import os
import sys
import torch
import torch.nn as nn
from component.svd_llama import LlamaForCausalLM


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id,troch_dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    if 'llama' in model_id.lower():
        model = LlamaForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=troch_dtype, trust_remote_code=True, cache_dir=None)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=troch_dtype, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    print(f'\n===\n[DEBUG]: model_id={model_id}')
    pruned_dict = torch.load(model_id, weights_only=False, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
