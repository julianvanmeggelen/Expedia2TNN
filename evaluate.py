import torch.nn.functional as F
import torch.nn as nn
import torch
from data import ValDataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

def batchPredictCosineSimilarity(X_query_cat, X_query_num, X_item_cat, X_item_num, model):
    e_q, e_i = model(X_query_cat, X_query_num, X_item_cat, X_item_num)
    similarity = F.cosine_similarity(e_q, e_i,  dim=1)
    return similarity

def valPrediction(model, valDataLoader, verbose=False):
    """
    return dataframe with relevance per srch_id, prop_id ordered by sim per srch_id
    """
    def convertIndex(index):
        return [tuple(_) for _ in index]
    
    res = pd.DataFrame(columns = ['index_srch_id', 'index_prop_id', 'sim', 'weight'], data=np.zeros(shape=(valDataLoader.nRecords, 4)))
    convertedIndex = convertIndex(valDataLoader.val_index)
    res[['index_srch_id', 'index_prop_id']] = convertedIndex
    res.index = convertedIndex
    for X_query_cat, X_query_num, X_item_cat, X_item_num , w, index in tqdm(valDataLoader, disable= (not verbose)):
        similarity = batchPredictCosineSimilarity(X_query_cat, X_query_num, X_item_cat, X_item_num , model)
        batchConvertedIndex = convertIndex(index)
        res.loc[batchConvertedIndex,'sim'] = similarity.detach().numpy()
        res.loc[batchConvertedIndex,'weight'] = w.detach().numpy()
    return res

def DCG(resultsDf, p=5, inplace=False, sim_column = 'sim'):
    dcg_df = None
    if inplace:
        dcg_df = resultsDf
    else:
        dcg_df = resultsDf.copy()
    dcg_df = dcg_df.sort_values(by=['index_srch_id', sim_column], ascending=False)
    dcg_df['rank'] = dcg_df.groupby(by=['index_srch_id', sim_column]).cumcount()+1
    dcg_df = dcg_df.groupby('index_srch_id').head(p) #only keep p best predictions
    dcg_df['dcg_partial'] = dcg_df['weight']/np.log2(dcg_df['rank']+1)
    dcg_vals = dcg_df[['index_srch_id','dcg_partial']].groupby('index_srch_id').sum()['dcg_partial']
    return dcg_vals.mean()

def valDcg(dataLoader, model, sim_column = 'sim'):
    model.eval()
    resultsDf = valPrediction(valDataLoader=dataLoader, model=model)
    dcg = DCG(resultsDf, sim_column=sim_column)
    model.train()
    return dcg

class DummyModel(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X_query_cat, X_query_num, X_item_cat, X_item_num):
        q = torch.normal(1,0,(X_query_cat.shape[0], self.embedding_dim))
        i = torch.normal(1,0,(X_query_cat.shape[0], self.embedding_dim))
        return q, i
   
def randomOrderingBenchmark(dataLoader):
    model = DummyModel()
    dcg = valDcg(dataLoader=dataLoader, model=model)
    return dcg

def perfectOrderingBenchmark(dataLoader):
    model = DummyModel()
    dcg = valDcg(dataLoader=dataLoader, model=model, sim_column='weight')
    return dcg





    
        