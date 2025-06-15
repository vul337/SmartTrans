import torch
from torch.utils.data import DataLoader
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import os
from typing import List, Optional, Tuple, Union
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from models import JtransEncoder
device_ids = [0, 1, 2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


WANDB = True
def tokenize_function(examples):
    # Redirect JMP_XXX 
    full_tokens = tokenizer(
        examples,
        truncation=True,
        max_length=args.max_seq_length,
        return_special_tokens_mask=True,
        padding='max_length'
    )
    dst_id = tokenizer('JUMPDEST')['input_ids'][0]
    line = full_tokens['input_ids']
    attention_mask = full_tokens['attention_mask']
    length = len(line)
    dst_lst = []
    for i, token in enumerate(line):
        if token == dst_id:
            dst_lst.append(i)
    for i, token in enumerate(line):
        if token < 4096 and token != i:
            if token < len(dst_lst):
                line[i] = dst_lst[token]
            else:
                line[i] = i
        
            
    return torch.LongTensor(line).to(dev), torch.LongTensor(attention_mask).to(dev)
    

def get_ebd(model, datas):
    with torch.no_grad():
        ebd = {}
        for addr in tqdm(datas.keys()):
            ebd[addr] = {}
            for func, data in datas[addr].items():
                input_ids, att_mask = tokenize_function(data)
                output = model(input_ids=input_ids.unsqueeze(dim=0), attention_mask=att_mask.unsqueeze(dim=0))
                ebd[addr][func] = output.cpu()
        
    return ebd

import torch

def all_get_ebd_parallel(model, datas, batch_size, device_ids):
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device_ids[0]) 
    with torch.no_grad():
        ebd = {}
        keys = list(datas.keys())
        
        for i in tqdm(range(0, len(keys), batch_size)):
            batch_files = keys[i:i+batch_size]
            input_ids_list = []
            att_mask_list = []
            contract_names = []
            funcnames = []
            
            for file in batch_files:
                contract_name = file.split('-')[0]
                funcname = file.split('-')[-1]
                if contract_name not in ebd.keys():
                    ebd[contract_name] = {}
                input_ids, att_mask = tokenize_function(datas[file])
                input_ids_list.append(input_ids)
                att_mask_list.append(att_mask)
                contract_names.append(contract_name)
                funcnames.append(funcname)
            
            input_ids_batch = torch.stack(input_ids_list)
            att_mask_batch = torch.stack(att_mask_list)

            input_ids_batch = input_ids_batch.to(device_ids[0]) 
            att_mask_batch = att_mask_batch.to(device_ids[0])
            
            output = model(input_ids=input_ids_batch, attention_mask=att_mask_batch)
            
            for i in range(len(batch_files)):
                ebd[contract_names[i]][funcnames[i]] = output[i].cpu().unsqueeze(0)
    return ebd
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SaveEmbedding")
    parser.add_argument("--model_path", type=str, default='./pretrain_model/epoch_1',  help='the path of pretrain model')
    parser.add_argument("--finetune_path", type=str, default='./finetune/512_margin03_5epoch/epoch_5/SmartTrans_model_state_dict.pt',  help='the path of finetune model')
    parser.add_argument("--tokenizer", type=str, default='./tokenizer', help='the path of tokenizer')
    parser.add_argument("--eval_batch_size", type=int, default = 512, help='evaluation batch size')
    parser.add_argument("--data_path", type=str, default='./data/preprocess_cfg.pkl', help='the path of training data')
    parser.add_argument("--save_path", type=str, default='./data/embeddings.pkl', help='the path of save embedding')

    parser.add_argument("--max_seq_length", type=int, default=512, help='max_seq_length')
    args = parser.parse_args()

    if torch.cuda.is_available():
        dev=torch.device('cuda')
    else:
        dev=torch.device('cpu')
    print(dev)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    model = JtransEncoder(
        dim=128,
        pretrain_path=args.model_path
    )
    model.load_state_dict(
        torch.load(args.finetune_path,
        map_location="cpu"),
    )
    
    model.to(dev)
    model.eval()
    
    with open(args.data_path, 'rb') as f:
        datas = pickle.load(f)

    ebds = get_ebd(model, datas)

    with open(args.save_path, 'wb') as f:
        pickle.dump(ebds, f)
        
        
    
