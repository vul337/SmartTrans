import pickle
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from collections import deque
import json
from myclient.facts_to_cfg import *
import os
import sys
import pandas as pd
import subprocess
import time
from tqdm import tqdm
import threading
import shutil

TEMP_PATH = '{YOUR PATH TO GIGAHORSE}/.temp' # [IO]
SAVE_PATH = './data' # [IO]
def dfs(node, succ, graph, func_name, all_asm):
    vis = {}
    q = deque()
    q.append((node, succ))
    while q:
        node, succ = q.pop()
        vis[node] = 1
        for suc in succ:
            my_asm = []
            for i, asm in enumerate(suc.statements):
                # if asm.op == 'CONST':
                #     # my_asm.append(f"PUSH{int(suc.statements[i+1].ident.split('0x')[1], 16)-int(suc.statements[i].ident.split('0x')[1], 16)-1} CONST")
                #     my_asm.append(asm.op)
                # else:
                addr = '0x' + asm.ident.split('0x')[1]
                if addr in all_asm.keys():
                    my_asm.append(all_asm[addr])
                else:
                    my_asm.append(asm.op)
            graph[func_name].add_node(suc.ident, asm=my_asm)
            graph[func_name].add_edge(node, suc.ident)
            if suc.ident in vis.keys():
                continue
            if suc.successors == []:
                continue
            q.append((suc.ident, suc.successors))

def extract_cfg(filename):
    target_path = os.path.join(TEMP_PATH, filename)
    _, functions = construct_cfg(f'{target_path}/out')
    if functions == 0:
        print('Wrong:', filename)
        return
    with open(target_path + '/contract.dasm', 'r') as f:
        contract_asm = f.readlines()
    with open(target_path + '/bytecode.hex', 'r') as f:
        btcode = f.read() 
    all_asm = {}
    for line in contract_asm:
        all_asm[line.split(':')[0].strip(' ')] = line.split(':')[1].strip('\n').strip(' ')
    graph = {}
    for nodes_key in functions.keys():
        func_name = functions[nodes_key].name
        graph[func_name] = nx.DiGraph()
        # node_addr = int(nodes_key, 16)
        my_asm = []
        for i, asm in enumerate(functions[nodes_key].head_block.statements):
            addr = '0x' + asm.ident.split('0x')[1]
            if addr in all_asm.keys():
                my_asm.append(all_asm[addr])
            else:
                my_asm.append(asm.op)
        graph[func_name].add_node(nodes_key, asm=my_asm)
        dfs(nodes_key, functions[nodes_key].head_block.successors, graph, func_name, all_asm)
    result = {'bytecode':btcode, 'asmcode':all_asm, 'graph':graph}
    return result


if __name__=='__main__':
    if not os.path.exists(TEMP_PATH):
        raise FileNotFoundError(f"Temporary path {TEMP_PATH} does not exist. Please check the path.")
    
    addresses = os.listdir(TEMP_PATH)
    results = {}
    for addr in tqdm(addresses):
        results[addr] = extract_cfg(addr)
    with open(os.path.join(SAVE_PATH, 'cfgs.pkl'), 'wb') as f:
        pickle.dump(results, f)
        

