import pickle
import os
import networkx as nx
from tqdm import tqdm
import threading
import re
import time
import logging
def remove_multiple_spaces(string):
    return re.sub(' +', ' ', string)

def playdata(result):
    results = {}
    for func, cfg in result['graph'].items():
        func_data = ''
        _addr = []
        full_addr = {}
        addr_to_jump_posi = {}
        for k in cfg.nodes.keys():
            addr = '0x' + k.split('0x')[1].strip('B')
            if addr not in _addr:
                _addr.append(int(addr, 16))
                full_addr[str(int(addr, 16))] = k
        _addr = sorted(_addr)
        
        jmp_pos = [0]
        for idx, node in enumerate(_addr):
            node_asm = cfg.nodes[full_addr[str(node)]]['asm']
            for inst in node_asm:
                func_data += remove_multiple_spaces(inst) + ' '
            jmp_lst = list(cfg._succ[full_addr[str(node)]].keys())
            if len(jmp_lst) > 0:
                jmp_tar = int('0x' + jmp_lst[0].split('0x')[1].strip('B'), 16)
                if len(jmp_lst) == 2:
                    jmp2 = int('0x' + jmp_lst[1].split('0x')[1].strip('B'), 16)
                    if jmp2 <= node:
                        jmp_tar = jmp2
                    if jmp2 > jmp_tar and jmp_tar > node:
                        jmp_tar = jmp2
                for i in range(len(_addr)):
                    if _addr[i] == jmp_tar:
                        break
                func_data += 'JUMP_' + str(i) + ' ' #The first is jump_0
            jmp_pos.append(len(func_data.strip(' ').split(' ')))
        results[func] = [func_data, jmp_pos]
    return results
def solve_position(func_items):
    # del position
    results = {}
    for func, item in func_items.items():
        line = item[0].split(' ')
        positions = item[1]
        dst = []
        for k, token in enumerate(line):
            if token.startswith('JUMP_'):
                j = int(token[5:])
                # line[k] = 'JUMP_' + str(positions[i][j])
                if positions[j] not in dst:
                    dst.append(positions[j])
        for d in sorted(dst, reverse=True):
            line.insert(d, 'JUMPDEST')
        dst = sorted(dst)
        for k, token in enumerate(line):
            if token.startswith('JUMP_'):
                plus = 0
                j = int(token[5:])
                for addr in dst:
                    if addr >= positions[j]:
                        break
                    plus += 1
                line[k] = 'JUMP_' + str(positions[j] + plus)
        results[func] = ' '.join(line)
    return results

def txt_to_block_length(func_items, block_length):
    results = {}
    for func, item in func_items.items():
        l = item.split(' ')
        dst_list = []
        for i, token in enumerate(l):
            if token == 'JUMPDEST':
                dst_list.append(i)
        block_num = (len(l)-1) // block_length + 1
        i = 0
        d = {}
        num = 0
        for j in dst_list:
            if j >= i * block_length and j < (i+1)*block_length:
                d[j] = num
                num += 1
        for i_512, token in enumerate(l[i*block_length:(i+1)*block_length]):
            if token.startswith('JUMP_'):
                j = int(token[5:])
                if j >= i * block_length and j < (i+1)*block_length:
                    l[i_512+i*block_length] = 'JUMP_' + str(d[j])
                else:
                    l[i_512+i*block_length] = 'JUMP_' + str(block_length+1)
        results[func] = ' '.join(l[i*block_length: (i+1)*block_length])
    return results

def add_cls(results):
    for func, _ in results.items():
        results[func] = '[CLS] ' + results[func]
    return results

def solve_const(result):
    addr_list = list(result['asmcode'].keys())
    for func, cfg in result['graph'].items():
        for _addr, bb in cfg.nodes.items():
            addr = '0x' + _addr.split('0x')[1].strip('B')
            if addr not in addr_list:
                continue
            for st in range(len(addr_list)):
                if addr_list[st] == addr:
                    break
            insts = []
            for x in range(st, len(addr_list)):
                inst = result['asmcode'][addr_list[x]]
                if inst == 'JUMP' or inst == 'JUMPI':
                    insts.append(inst)
                    break
                if inst == 'JUMPDEST' and x==st:
                    continue
                if inst == 'JUMPDEST' and x!=st:
                    break
                insts.append(inst)
            if len(insts) == 0:
                continue
            bb['asm'] = insts
    return result
def preprocess(disasm_graph, block_length=512):
    disasm_graph2 = solve_const(disasm_graph)
    data1 = playdata(disasm_graph2)
    tokens = solve_position(data1)
    result1 = txt_to_block_length(tokens, block_length)
    result2 = add_cls(result1)
    return result2
if __name__=='__main__':
    CFG_FILE = './data/cfgs.pkl' # [IO]
    OUT_FILE = './data/preprocess_cfg.pkl' # [IO]

    results = {}
    with open(CFG_FILE, 'rb') as f:
        cfgs = pickle.load(f)
    for addr, cfg in tqdm(cfgs.items()):
        results[addr] = preprocess(cfg)
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    
