import ml_collections as mlc
import torch.nn as nn
from data import QUERY_CAT_FEATURE_COLS, QUERY_NUM_FEATURE_COLS, ITEM_CAT_FEATURE_COLS, ITEM_NUM_FEATURE_COLS, ITEM_NUM_INDICATOR_COLS, QUERY_NUM_INDICATOR_COLS, getMappings, ADD_INDICATOR
defaultCatEmbeddingDim = 8


maps_to_ind, _ = getMappings(train=True)

#Default model config
queryTowerCfg = mlc.ConfigDict()
queryTowerCfg.embedding_dim = [defaultCatEmbeddingDim] * len(QUERY_CAT_FEATURE_COLS)
#queryTowerCfg.embedding_dict_size = [len(v)+1 for k,v in maps_to_ind.items() if k in QUERY_CAT_FEATURE_COLS]
queryTowerCfg.embedding_dict_size = [len(maps_to_ind[k])+1 for k in QUERY_CAT_FEATURE_COLS if k in maps_to_ind.keys()]

queryTowerCfg.numeric_dim = [len(QUERY_NUM_FEATURE_COLS)+(len(QUERY_NUM_INDICATOR_COLS) if ADD_INDICATOR else 0), 8, 16]
queryTowerCfg.shared_hidden_dim = [128, 64, 32]
queryTowerCfg.activation = nn.ReLU()
queryTowerCfg.dropout =0.3
queryTowerCfg.useAttention = False

itemTowerCfg = mlc.ConfigDict()
itemTowerCfg.embedding_dim = [defaultCatEmbeddingDim] * len(ITEM_CAT_FEATURE_COLS)
#itemTowerCfg.embedding_dict_size = [len(v)+1 for k,v in maps_to_ind.items() if k in ITEM_CAT_FEATURE_COLS]
itemTowerCfg.embedding_dict_size = [len(maps_to_ind[k])+1 for k in ITEM_CAT_FEATURE_COLS if k in maps_to_ind.keys()]

itemTowerCfg.numeric_dim = [len(ITEM_NUM_FEATURE_COLS)+(len(ITEM_NUM_INDICATOR_COLS)if ADD_INDICATOR else 0), 20, 16]
itemTowerCfg.shared_hidden_dim = [84, 64, 32]
itemTowerCfg.activation = nn.ReLU()
itemTowerCfg.dropout = 0.3
itemTowerCfg.useAttention = False

modelCfg = mlc.ConfigDict()
modelCfg.embedding_dim = 18
modelCfg.itemTowerCfg = itemTowerCfg
modelCfg.queryTowerCfg = queryTowerCfg

# Training config
trainCfg = mlc.ConfigDict() 
trainCfg.n_epoch = 100
trainCfg.batch_size = 196608
trainCfg.lr = 0.0003
trainCfg.negFrac = 0.4
trainCfg.crossFrac = 0.3


if __name__ == "__main__":
    print("="*30, " Default configurations", "="*30)
    print("-"*30, " Query tower ", "-"*30)
    print(queryTowerCfg)
    print("-"*30, " Item tower ", "-"*30)
    print(itemTowerCfg)
    print("-"*30, " Model ", "-"*30)
    print(modelCfg)
    print("-"*30, " Train params ", "-"*30)
    print(trainCfg)