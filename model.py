import torch
import torch.nn as nn
import torch.nn.functional as F
from config import modelCfg as defaultModelCfg

class Tower(nn.Module):
    def __init__(self, embedding_dim: list, embedding_dict_size:list, numeric_dim:list, shared_hidden_dim:list, output_embedding_dim, activation=nn.ReLU, dropout=None, useAttention=False):
        super().__init__()
        self.useAttention = useAttention
        self.dropout = dropout
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=s, embedding_dim=d) for (d,s) in zip(embedding_dim, embedding_dict_size)])
        nCatFeatures = len(embedding_dim)
        self.activation = activation
        #Numeric feed forward
        self.numeric_ffw = nn.ModuleList([nn.Linear(numeric_dim[i],numeric_dim[i+1]) for i in range(len(numeric_dim)-1)])

        #Shared ffw (after concat)
        shared_ffw_inputsize = sum(embedding_dim) + numeric_dim[-1]
        self.shared_ffw = nn.ModuleList([nn.Linear(shared_ffw_inputsize, shared_hidden_dim[0])])
        self.shared_ffw.extend([nn.Linear(shared_hidden_dim[i],shared_hidden_dim[i+1]) for i in range(len(shared_hidden_dim)-1)])
        self.shared_ffw.append(nn.Linear(shared_hidden_dim[-1], output_embedding_dim))

        self.num_batchnorm = nn.BatchNorm1d(num_features=numeric_dim[0])

        #Attention
        if useAttention:
            num_heads=  1
            self.att = nn.MultiheadAttention(shared_ffw_inputsize, num_heads, batch_first=True)

    def forward(self, X_cat, X_num):
        """
        X_query_cat: (batch_size, n_cat_features)
        X_query_num: (batch_size, n_num_features)

        Create embeddings for all cat features, concat with numeric features and do MLP to obtain final embedding
        """
        embeddings = []
        for i, l in enumerate(self.embeddings):
            emb = l(X_cat[:,i])
            embeddings.append(emb)
        num_out = X_num
        #num_out = self.num_batchnorm(X_num)
        for i, l in enumerate(self.numeric_ffw):
            num_out = l(num_out)
            num_out = self.activation(num_out) 

        x = torch.cat(embeddings + [num_out], dim=1)

        if self.useAttention:
            x,_ = self.att(x,x,x, need_weights=False)

        if self.dropout:
            x = nn.Dropout(p=self.dropout)(x)

        query_embedding_out = x
        for i, l in enumerate(self.shared_ffw):
            query_embedding_out = l(query_embedding_out)
            query_embedding_out = self.activation(query_embedding_out) 

        return query_embedding_out

class RecModel(nn.Module):
    def __init__(self, embedding_size, query_tower_args, item_tower_args):
        """
        embedding_size: final size of the two embedding
        query_tower_args & item_tower_args: dict with keys   embedding_dim, embedding_dict_size, numeric_dim, shared_hidden_dim, activation
        """
        super().__init__()
        query_tower_args['output_embedding_dim'] = embedding_size
        item_tower_args['output_embedding_dim'] = embedding_size

        self.query_tower = Tower(**query_tower_args)
        self.item_tower = Tower(**item_tower_args)
    
    def forward(self, X_query_cat, X_query_num, X_item_cat, X_item_num):
        q = self.query_tower(X_query_cat, X_query_num)
        i = self.item_tower(X_item_cat, X_item_num)
        return q, i

def getDefaultModel(modelCfg = None):
    if modelCfg is None:
        modelCfg = defaultModelCfg
    return RecModel(modelCfg.embedding_dim, query_tower_args=dict(modelCfg.queryTowerCfg), item_tower_args = dict(modelCfg.itemTowerCfg))

# def weightedCoSim(w, q, i):
#     """
#     w: (batch_size)
#     q: (batch_size, embedding_dim)
#     i: (batch_size, embedding_dim)
#     """
#     return torch.mean(-w * F.cosine_similarity(q, i, dim=1))


def weightedCoSim(w, q, i):
    """
    w: (batch_size)
    q: (batch_size, embedding_dim)
    i: (batch_size, embedding_dim)
    """
    y = torch.where(w>0,1,-1)
    loss  = F.cosine_embedding_loss(q, i, w, reduction='mean')
    #w_use = torch.where(w>1,5,1)
    return loss#torch.mean(w_use * loss)

      

