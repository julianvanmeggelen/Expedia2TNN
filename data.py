import pandas as pd
import numpy as np
import os 
from collections import defaultdict
import pickle 
import torch


MAPS_TO_IND_PATH = 'maps_to_ind.pkl'
MAPS_FROM_IND_PATH = 'maps_from_ind.pkl'
TRAIN_SET_PATH = 'train_set.pkl'

TRAIN_QUERY_ARRAY_PATH = 'train_query.npy'
TRAIN_ITEM_ARRAY_PATH = 'train_item.npy'
TRAIN_REL_ARRAY_PATH = 'train_rel.npy'

QUERY_NUM_FEATURE_COLS = [ #Cat columns first! 
    'visitor_hist_starrating'
    ,'visitor_hist_adr_usd'
    ,'srch_query_affinity_score'
    ,'orig_destination_distance'
]


QUERY_NUM_INDICATOR_COLS = [
    'orig_destination_distance_isNaN'
    ,'srch_query_affinity_score_isNaN'
    ,'visitor_hist_adr_usd_isNaN'
    ,'visitor_hist_starrating_isNaN'
]


QUERY_CAT_FEATURE_COLS = [ #Cat columns first! 
    'srch_id'
    ,'site_id'
    ,'visitor_location_country_id'
    ,'srch_length_of_stay'
    ,'srch_booking_window'
    ,'srch_adults_count'
    ,'srch_children_count'
    ,'srch_room_count'
    ,'srch_destination_id'
    ,'srch_saturday_night_bool'
    ,'random_bool'
]

QUERY_FEATURE_COLS = QUERY_CAT_FEATURE_COLS + QUERY_NUM_FEATURE_COLS + QUERY_NUM_INDICATOR_COLS

ITEM_NUM_FEATURE_COLS = [
    'prop_review_score'
    ,'prop_location_score1'
    ,'prop_location_score2'
    ,'price_usd'
    ,'comp1_rate_percent_diff'
    ,'comp2_rate'
    ,'comp2_inv'
    ,'comp2_rate_percent_diff'
    ,'comp3_rate'
    ,'comp3_inv'
    ,'comp3_rate_percent_diff'
    ,'comp4_rate'
    ,'comp4_inv'
    ,'comp4_rate_percent_diff'
    ,'comp5_rate'
    ,'comp5_inv'
    ,'comp5_rate_percent_diff'
    ,'comp6_rate'
    ,'comp6_inv'
    ,'comp6_rate_percent_diff'
    ,'comp7_rate'
    ,'comp7_inv'
    ,'comp7_rate_percent_diff'
    ,'comp8_rate'
    ,'comp8_inv'
    ,'comp8_rate_percent_diff'
    ,'prop_log_historical_price'
]

ITEM_NUM_INDICATOR_COLS = [
    'comp1_rate_percent_diff_isNaN'
    ,'comp2_inv_isNaN'
    ,'comp2_rate_isNaN'
    ,'comp2_rate_percent_diff_isNaN'
    ,'comp3_inv_isNaN'
    ,'comp3_rate_isNaN' 
    ,'comp3_rate_percent_diff_isNaN'
    ,'comp4_inv_isNaN'
    ,'comp4_rate_isNaN'
    ,'comp4_rate_percent_diff_isNaN'
    ,'comp5_inv_isNaN'
    ,'comp5_rate_isNaN'
    ,'comp5_rate_percent_diff_isNaN'
    ,'comp6_inv_isNaN'
    ,'comp6_rate_isNaN'
    ,'comp6_rate_percent_diff'
    ,'comp6_rate_percent_diff_isNaN'
    ,'comp7_inv_isNaN'
    ,'comp7_rate_isNaN'
    ,'comp7_rate_percent_diff_isNaN'
    ,'comp8_inv_isNaN'
    ,'comp8_rate_isNaN'
    ,'comp8_rate_percent_diff_isNaN'
    ,'prop_location_score2_isNaN'
    ,'prop_review_score_isNaN'
]

ITEM_CAT_FEATURE_COLS = [
    'prop_country_id'
    ,'prop_id'
    ,'prop_starrating'
    ,'prop_brand_bool'
    ,'promotion_flag'
    ,'comp1_rate'
    ,'comp1_inv'
]

ITEM_FEATURE_COLS = ITEM_CAT_FEATURE_COLS + ITEM_NUM_FEATURE_COLS + ITEM_NUM_INDICATOR_COLS


CAT_FEATURE_COLS = QUERY_CAT_FEATURE_COLS + ITEM_CAT_FEATURE_COLS


def returnZero():
        return 0
def returnNone():
        return None

def getMappings(df_train=None, train=True, useCached = True):

    if os.path.exists(MAPS_TO_IND_PATH) and useCached:
        with open(MAPS_TO_IND_PATH, 'rb') as handle:
            maps_to_ind = pickle.load(handle)
        with open(MAPS_FROM_IND_PATH, 'rb') as handle:
            maps_from_ind = pickle.load(handle)
        return maps_to_ind, maps_from_ind
    
    if train:
        maps_to_ind = {}
        maps_from_ind = {}
        for col in CAT_FEATURE_COLS:
            unique = np.sort(df_train[col].unique().astype('int'))
            map_from_ind = {i+1:_ for i,_ in enumerate(unique)} #0 is the default for unknown id's
            map_to_ind = {_:i+1 for i,_ in enumerate(unique)}
            maps_from_ind[col] = defaultdict(returnNone,map_from_ind)
            maps_to_ind[col] = defaultdict(returnZero,map_to_ind)

        with open(MAPS_FROM_IND_PATH, 'wb') as handle:
            pickle.dump(maps_from_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(MAPS_TO_IND_PATH, 'wb') as handle:
            pickle.dump(maps_to_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)  
  
        return maps_to_ind, maps_from_ind
    else:
        return Exception("Mapping cannot be loaded, first needs to be generated using train=True")


def addNaNIndicator(df_in, inplace=True):
    if not inplace:
        df = df_in.copy()
    else:
        df = df_in

    res = {}
    for col in df.columns:
        nan = df[col].isna()
        if np.sum(nan)>0:
            df[f'{col}_isNaN'] = nan.astype('int64')
            res[col] = f'{col}_isNaN'
    df=fillNaN(df)
    return df, res

def fillNaN(df_in):
    df =df_in
    #Simple imputation
    for i in df.columns[df.isna().any(axis=0)]:     #---Applying Only on variables with NaN values
        df[i].fillna(df[i].mean(),inplace=True)
    return df

def applyMapping(df_in, mappings, inplace=True):
    if not inplace:
        df = df_in.copy()
    else:
        df = df_in
    for col, mapping in mappings.items():
        df[col] = df[col].apply(lambda val: mapping[val])
    return df
        
def getTrainData(useCached = True):
    if os.path.exists(TRAIN_SET_PATH) and useCached:
        with open(TRAIN_SET_PATH, 'rb') as handle:
            df_train = pickle.load(handle)
            print("Loaded train_data from disk")
            return df_train
    else:        
        df_train = pd.read_csv('training_set_VU_DM.csv')
        maps_to_ind, maps_from_ind = getMappings(df_train, train=True)
        df_train = applyMapping(df_train, maps_to_ind, inplace=True)
        df_train, nanIndicColumns = addNaNIndicator(df_train)
        #df_train = df_train[CAT_FEATURE_COLS + list(df_train.columns.difference(CAT_FEATURE_COLS))] # move cat columns to the front
        with open(TRAIN_SET_PATH, 'wb') as handle:
            pickle.dump(df_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return df_train, nanIndicColumns

def getTrainArrays(useCached =True):
    if os.path.exists(TRAIN_QUERY_ARRAY_PATH) and useCached:
        df_train_query = np.load(TRAIN_QUERY_ARRAY_PATH)
        df_train_item  = np.load(TRAIN_ITEM_ARRAY_PATH)
        df_train_rel  = np.load(TRAIN_REL_ARRAY_PATH)
        return df_train_query, df_train_item, df_train_rel
    else:
        df_train, nanIndicColumns = getTrainData(useCached)
        df_train_query = df_train[QUERY_FEATURE_COLS].values
        df_train_item  = df_train[ITEM_FEATURE_COLS].values
        df_train_rel = np.where(df_train['booking_bool'] == 0, df_train['click_bool'], df_train['booking_bool'])
        np.save(arr= df_train_query, file=TRAIN_QUERY_ARRAY_PATH)
        np.save(arr= df_train_item, file=TRAIN_ITEM_ARRAY_PATH)
        np.save(arr= df_train_rel, file=TRAIN_REL_ARRAY_PATH)
        return df_train_query, df_train_item, df_train_rel

class TrainDataLoader():
    def __init__(self, batch_size, negFrac, crossFrac):
        self.train_query, self.train_item, self.train_rel = getTrainArrays(useCached=True)
        self.batch_size = batch_size
        self.pos_ind = np.argwhere(self.train_rel>0)[:,0]
        self.neg_ind = np.argwhere(self.train_rel==0)[:,0]
        self.all_ind = list(range(self.train_item.shape[0]))
        self.nNeg = int(negFrac * batch_size)
        self.nCross = int(crossFrac * batch_size)
        self.nPos = batch_size - self.nNeg - self.nCross
        self.queryNCat = len(QUERY_CAT_FEATURE_COLS)
        self.itemNCat = len(ITEM_CAT_FEATURE_COLS)

    def _samplePositive(self, n):
        ind_sample = np.random.choice(self.pos_ind, size=(n))
        weights= self.train_rel[ind_sample]
        return ind_sample, weights

    def _sampleNegative(self, n):
        weights= np.zeros(shape=(n))-1
        ind_sample = np.random.choice(self.neg_ind, size=(n))
        return ind_sample, weights


    def _crossSampleNegative(self, n):
        ind1 =  np.random.choice(self.all_ind, size=(n))
        ind2 =  np.random.choice(list(set(self.all_ind).difference(set(ind1))), size=(n)) #make sure we do not sample two indices that are similar
        weights= np.zeros(shape=(n))-1
        return ind1, ind2, weights

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.__len__(): raise StopIteration
        res = self.__getitem__(self.i)
        self.i += 1
        return res
    
    def __len__(self):
        return int(self.train_item.shape[0]/self.batch_size)
    
    def __getitem__(self, i):
        """
        returns:
            query_cat: (batch_size, nCatQueryFeatures)
            query_num: (batch_size, nNumQueryFeatures)
            item_cat: (batch_size, nCatItemFeatures)
            item_num: (batch_size, nNumItemFeatures)
            w: (batch_size, 1)
                -1 if unrelated 1 if clicked, 5 if booked
        """
        indPos, wPos = self._samplePositive(self.nPos)
        indNeg, wNeg = self._sampleNegative(self.nNeg)
        ind1Cross, ind2Cross, wCross = self._crossSampleNegative(self.nCross)
        querySampleInd = np.concatenate([indPos, indNeg, ind1Cross])       
        itemSampleInd = np.concatenate([indPos , indNeg , ind2Cross])
        w = np.hstack([wPos,wNeg,wCross])
        querySample = self.train_query[querySampleInd]
        itemSample = self.train_item[itemSampleInd]

        X_query_cat, X_query_num, X_item_cat, X_item_num = querySample[:,:self.queryNCat], querySample[:,self.queryNCat:], itemSample[:,:self.itemNCat], itemSample[:,self.itemNCat:]
        X_query_cat, X_query_num, X_item_cat, X_item_num = torch.Tensor(X_query_cat).long(), torch.Tensor(X_query_num).float(), torch.Tensor(X_item_cat).long(), torch.Tensor(X_item_num).float()
        w = torch.Tensor(w).float()
        w.requires_grad_(False)
        return  X_query_cat, X_query_num, X_item_cat, X_item_num , w
        


    

    



   
