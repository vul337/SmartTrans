import faiss
import pickle
import torch
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import json

def search(queries, index, ebd_list, top_k=200, shreshold=0.7):
    distance, preds = index.search(queries, k=top_k)
    
    results = []
    for idx in preds[0]:
        if distance[0][idx] < shreshold:
            continue
        results.append(ebd_list[idx])
        
    return results
# you can use the following function to create an index from embeddings
def index_create(xb, dim=128, measure=faiss.METRIC_INNER_PRODUCT, param='HNSW64'): 
    index = faiss.index_factory(dim, param, measure)  
    print(index.is_trained)                          # 此时输出为True 
    index.add(xb)
    faiss.write_index(index, './index/my_index.index')

def normalize(x):
    return x / torch.norm(x)

def ebd_to_list(ebd):
    xb = []
    ebd = normalize(ebd)
    xb.append(ebd.cpu().numpy().squeeze())
    return np.array(xb)

if __name__=='__main__':
    EBDS_PATH = './data/embeddings.pkl' # [IO]
    INDEX_PATH = './index/my_index.index' # [IO]
    INDEX_EBD_LIST_PATH = './index/DUP_solve_Dayu_finetune_ebds_epoch_5_margin03_ebd_list.pkl' # [IO]
    SAVE_PATH = './data/search_results.pkl' # [IO]
     
    with open(EBDS_PATH, 'rb') as f: 
        ebds = pickle.load(f)

    
    # random select one ebd to search
    for addr, _ in ebds.items():
        for func, ebd in _.items():
            xb = ebd_to_list(ebd)
            break
        break

    start_time = time.time()
    mid = time.time()

    index = faiss.read_index(INDEX_PATH)
    with open(INDEX_EBD_LIST_PATH, 'rb') as f:
        all_data_ebd_list = pickle.load(f)
    load_t = time.time()
    
    results = search(xb, index, all_data_ebd_list)
    
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(results, f)
    
    
    
    
    


