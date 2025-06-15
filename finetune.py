import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import wandb
import logging
import datasets
import pickle
import transformers
import numpy as np
import logging
from tqdm import tqdm
import random
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import List, Optional, Tuple, Union
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModel,
    BertModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from models import JtransEncoder
logger = get_logger(__name__)

WANDB = True
def tokenize_function(examples):
    # Redirect JMP_XXX 
    full_tokens = tokenizer(
        examples,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
        return_special_tokens_mask=True,
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
        
            
    return  torch.LongTensor(line), torch.LongTensor(attention_mask)
class FunctionDataset(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,datas):  #random visit
        self.datas = datas
    
                
    def __getitem__(self, idx):             #also return bad pair
        funcname=self.datas[idx] # (num_opt, dim_graph_features, dim_graph_edges, dim_adj_matrix)
        '''random select two opts of one function as the positive pair'''
        opt1 = random.choice(name_list[funcname])
        opt2 = opt1
        while opt1 == opt2:
            opt2 = random.choice(name_list[funcname])
        token_seq1, mask1 = tokenize_function(all_datas[opt1])
        token_seq2, mask2 = tokenize_function(all_datas[opt2])

        '''random select another function with opt as the negative sample'''
        neg_idx = random.randint(0, len(self.datas)-1)
        while neg_idx==idx:
            neg_idx = random.randint(0, len(self.datas)-1)
        funcname = self.datas[neg_idx]
        opt3 = random.choice(name_list[funcname])
        token_seq3, mask3 = tokenize_function(all_datas[opt3])

        return token_seq1, token_seq2, token_seq3, mask1, mask2, mask3
    def __len__(self):
        return len(self.datas)

def train_dp(model, args, train_dataloader, valid_dataloader, logger):

    class Triplet_COS_Loss(nn.Module):
        def __init__(self,margin):
            super(Triplet_COS_Loss, self).__init__()
            self.margin=margin

        def forward(self, repr, good_code_repr, bad_code_repr):
            good_sim=F.cosine_similarity(repr, good_code_repr)
            bad_sim=F.cosine_similarity(repr, bad_code_repr)
            #print("simm ",good_sim.shape)
            loss=(self.margin-(good_sim-bad_sim)).clamp(min=1e-6).mean()
            return loss

    # model = nn.DataParallel(model)
    global_steps = 0
    train_total_loss = 0
    for epoch in range(args.epoch):
        model.train()
        triplet_loss=Triplet_COS_Loss(margin=args.margin)
        # train_iterator = tqdm(train_dataloader)
        loss_list = []
        for i, (token_seq1, token_seq2, token_seq3, mask1, mask2, mask3) in enumerate(train_dataloader): 
            with accelerator.accumulate(model):
                anchor,pos,neg=0,0,0

                anchor = model(input_ids=token_seq1, attention_mask=mask1)

                pos = model(input_ids=token_seq2, attention_mask=mask2)

                neg = model(input_ids=token_seq3, attention_mask=mask3)

                loss = triplet_loss(anchor, pos, neg)
                loss.requires_grad_(True)

                # loss.backward()
                loss_list.append(loss.detach().float())
                train_total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (i+1) % 1 == 0:
                global_steps += 1
                # tmp_lr = optimizer.param_groups[0]["lr"]
                # logger.info(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], lr={tmp_lr}, loss={loss}")
                # train_iterator.set_description(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], lr={tmp_lr}, loss={loss}")
                if WANDB:
                    accelerator.log(
                            {
                                "step_lr": float(lr_scheduler.get_last_lr()[0]),
                                "train_step_loss": train_total_loss / 10,
                            }
                        )
                train_total_loss = 0
            if accelerator.sync_gradients:
                progress_bar.update(1)


        
        if (epoch+1) % args.save_every == 0:
            logger.info(f"Saving Model ...")
            output_dir = os.path.join(args.output_path, f"finetune_epoch_{epoch+1}")
            accelerator.wait_for_everyone()
            accelerator.save_state(f"{output_dir}/checkpoint-{epoch}")
            unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(
            #     output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            # )
            torch.save(
                unwrapped_model.state_dict(),
                f"{output_dir}/checkpoint-{epoch}/jtrans_model_state_dict.pt",
            )
            logger.info(f"Done")
        if (epoch+1) % args.eval_every == 0:
            logger.info(f"Doing Evaluation ...")
            mrr = finetune_eval(model, valid_dataloader)
            logger.info(f"[*] epoch: [{epoch}/{args.epoch+1}], mrr={mrr}")
            if WANDB:
                accelerator.log({
                'mrr': mrr
            })
def finetune_eval(net, data_loader):
    net.eval()
    with torch.no_grad():
        avg=[]
        gt=[]
        cons=[]
        SIMS=[]
        AP_1=[]
        AP_5=[]
        for i, (token_seq1, token_seq2, _, mask1, mask2, _) in enumerate(data_loader):

            anchor,pos=0,0

            anchor = model(input_ids=token_seq1, attention_mask=mask1)

            pos = model(input_ids=token_seq2, attention_mask=mask2)

            ans=0
            for i in range(len(anchor)):    # check every vector of (vA,vB)
                vA=anchor[i:i+1].cpu()  #pos[i]
                sim=[]
                for j in range(len(pos)):
                    vB=pos[j:j+1].cpu()   # pos[j]
                    AB_sim=F.cosine_similarity(vA, vB).item()
                    sim.append(AB_sim)
                    if j!=i:
                        cons.append(AB_sim)
                sim=np.array(sim)
                y=np.argsort(-sim)
                posi=0
                for j in range(len(pos)):
                    if y[j]==i:
                        posi=j+1

                gt.append(sim[i])

                ans+=1/posi

            ans=ans/len(anchor)
            avg.append(ans)
        return np.mean(np.array(avg))
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SmartTrans-Finetune")
    parser.add_argument("--model_path", type=str, default='./pretrain_model/epoch_1',  help='the path of pretrain model')
    parser.add_argument("--output_path", type=str, default='./finetune/512_margin03_5epoch', help='the path where the finetune model be saved')
    parser.add_argument("--tokenizer", type=str, default='./tokenizer', help='the path of tokenizer')
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True, help='low_cpu_mem_usage')
    parser.add_argument("--model_type", type=str, default='roformer', help='model_type')
    parser.add_argument("--epoch", type=int, default=5, help='number of training epochs')
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--warmup", type=int, default=1000, help='warmup steps')
    parser.add_argument("--step_size", type=int, default=40000, help='scheduler step size')
    parser.add_argument("--gamma", type=float, default=0.99, help='scheduler gamma')
    parser.add_argument("--batch_size", type=int, default = 16, help='training batch size')
    parser.add_argument("--eval_batch_size", type=int, default = 128, help='evaluation batch size')
    parser.add_argument("--log_every", type=int, default =1, help='logging frequency')
    parser.add_argument("--local_rank", type=int, default = 0, help='local rank used for ddp')
    parser.add_argument("--freeze_cnt", type=int, default=10, help='number of layers to freeze')
    parser.add_argument("--weight_decay", type=int, default = 1e-4, help='regularization weight decay')
    parser.add_argument("--eval_every", type=int, default=1, help="evaluate the model every x epochs")
    parser.add_argument("--eval_every_step", type=int, default=1000, help="evaluate the model every x epochs")
    parser.add_argument("--save_every", type=int, default=1, help="save the model every x epochs")
    parser.add_argument("--data_path", type=str, default='./finetune_dataset/512_datas.pkl', help='the path of training data')
    parser.add_argument("--name_list_path", type=str, default='./finetune_dataset/name_list.pkl', help='the path of name list')
    parser.add_argument("--max_seq_length", type=int, default=512, help='max_seq_length')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help='gradient_accumulation_steps')
    parser.add_argument("--max_train_steps", type=int, default=None, help='max_train_steps')
    parser.add_argument("--report_to", type=str, default="wandb", help='report_to')
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help='preprocessing_num_workers')
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup", help="The scheduler type to use.")
    parser.add_argument("--num_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet Loss Margin.")

    args = parser.parse_args()
    

    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_path

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = JtransEncoder(
        dim=128,
        pretrain_path=args.model_path
    )
    
    freeze_layer_count = args.freeze_cnt
    for param in model.parameters():
        param.requires_grad = False

    if freeze_layer_count != -1:
        for layer in model.encoder.encoder.layer[:freeze_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # print(optimizer.param_groups)
    for i, param_group in enumerate(optimizer.param_groups):
        for param in param_group['params']:
            param.requires_grad_(True)
    # print(optimizer.param_groups)
    
    
    with open(args.name_list_path, 'rb') as f:
        name_list = pickle.load(f)
    with open(args.data_path, 'rb') as f:
        all_datas = pickle.load(f)
    test_list = list(name_list.keys())[-len(name_list)//5:]
    train_list = list(name_list.keys())[:-len(name_list)//5]
    # test_list = list(name_list.keys())[-128*4:]
    # train_list = list(name_list.keys())[:64]
    test_loader = DataLoader(FunctionDataset(test_list), batch_size=args.eval_batch_size, num_workers=args.preprocessing_num_workers, shuffle=True, drop_last=True)
    train_loader = DataLoader(FunctionDataset(train_list), batch_size=args.batch_size, num_workers=args.preprocessing_num_workers, shuffle=True, drop_last=True)
    # print(optimizer.param_groups)
    # for i, param_group in enumerate(optimizer.param_groups):
    #     clf = [
    #         param for param in param_group['params'] if param.requires_grad
    #         ]
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    print("num_warmup_steps:", args.num_warmup_steps * args.gradient_accumulation_steps)
    print("num_training_steps:", args.max_train_steps * args.gradient_accumulation_steps)
    # print("after_data_loader:", len(train_dataloader))
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
    # for i, (token_seq1, token_seq2, token_seq3, mask1, mask2, mask3) in enumerate(train_loader):
    #     check = model(input_ids=token_seq1, attention_mask=mask1)
    #     print(1)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.epoch * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.epoch = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("jTrans_finetune_DDP", experiment_config)
    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0
    
    train_dp(model, args, train_loader, test_loader, logger)
