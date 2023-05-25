import torch.nn.functional as F
from data import ValDataLoader
import pandas as pd
import numpy as np

def batchPredictCosineSimilarity(X_query_cat, X_query_num, X_item_cat, X_item_num, model):
    e_q, e_i = model(X_query_cat, X_query_num, X_item_cat, X_item_num)
    similarity = F.cosine_similarity(e_q, e_i,  dim=1)
    return similarity

def valPrediction(model, valDataLoader):
    """
    return dataframe with relevance per srch_id, prop_id ordered by sim per srch_id
    """
    def convertIndex(index):
        return [tuple(_) for _ in index]
    
    res = pd.DataFrame(columns = ['srch_id', 'prop_id', 'sim', 'weight'], data=np.zeros(shape=(valDataLoader.nRecords, 4)))
    convertedIndex = convertIndex(valDataLoader.val_index)
    res[['srch_id', 'prop_id']] = convertedIndex
    res.index = convertedIndex
    for X_query_cat, X_query_num, X_item_cat, X_item_num , w, index in valDataLoader:
        similarity = batchPredictCosineSimilarity(X_query_cat, X_query_num, X_item_cat, X_item_num , model)
        batchConvertedIndex = convertIndex(index)
        res.loc[batchConvertedIndex]['sim'] = similarity.detach().numpy()
        res.loc[batchConvertedIndex]['weight'] = w
    res = res.sort_values(by=['srch_id', 'sim'], ascending=False)
    res['rank'] = res.groupby(by=['srch_id', 'sim']).cumcount()+1
    return res

def DCG(resultsDf, p=10, inplace=False):
    dcg_df = None
    if inplace:
        dcg_df = resultsDf
    else:
        dcg_df = resultsDf.copy()
    dcg_df = dcg_df.groupby('srch_id').head(p) #only keep p best predictions
    dcg_df['dcg_partial'] = dcg_df['weight']/np.log2(dcg_df['rank']+1)
    dcg_vals = dcg_df[['srch_id','dcg_partial']].groupby('srch_id').sum()['dcg_partial']
    return dcg_vals.mean()

def valDcg(dataLoader, model):
    model.eval()
    resultsDf = valPrediction(valDataLoader=dataLoader, model=model)
    dcg = DCG(resultsDf)
    model.train()
    return dcg



#def randomOrderingBenchmark()
        