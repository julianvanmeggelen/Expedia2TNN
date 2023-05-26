import pandas as pd
import numpy as np
import os 
from collections import defaultdict
import pickle 
import torch
from tqdm import tqdm
tqdm.pandas()


N_VAL_RECORDS = 500000

DATA_DIR = './data/'
RAW_TRAIN_DATA_PATH = DATA_DIR+ 'training_set_raw.csv'
RAW_VAL_DATA_PATH = DATA_DIR+'val_set_raw.csv'
RAW_TEST_DATA_PATH = 'test_set_VU_DM.csv'

MAPS_TO_IND_PATH = DATA_DIR+'maps_to_ind.pkl'
MAPS_FROM_IND_PATH = DATA_DIR+'maps_from_ind.pkl'

TRAIN_SET_PATH = DATA_DIR+'train_set.pkl'
VAL_SET_PATH = DATA_DIR+'val_set.pkl'
TEST_SET_PATH = DATA_DIR + 'test_set.pkl'

TRAIN_INDEX_ARRAY_PATH = DATA_DIR+'train_index.npy'
TRAIN_QUERY_ARRAY_PATH = DATA_DIR+'train_query.npy'
TRAIN_ITEM_ARRAY_PATH = DATA_DIR+'train_item.npy'
TRAIN_REL_ARRAY_PATH = DATA_DIR+'train_rel.npy'

VAL_INDEX_ARRAY_PATH = DATA_DIR+'val_index.npy'
VAL_QUERY_ARRAY_PATH = DATA_DIR+'val_query.npy'
VAL_ITEM_ARRAY_PATH = DATA_DIR+'val_item.npy'
VAL_REL_ARRAY_PATH = DATA_DIR+'val_rel.npy'

TEST_INDEX_ARRAY_PATH = DATA_DIR+'test_index.npy'
TEST_QUERY_ARRAY_PATH = DATA_DIR+'test_query.npy'
TEST_ITEM_ARRAY_PATH = DATA_DIR+'test_item.npy'
TEST_REL_ARRAY_PATH = DATA_DIR+'test_rel.npy'

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
    #,'comp1_rate'
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
    #,'comp1_inv'
]

ITEM_FEATURE_COLS = ITEM_CAT_FEATURE_COLS + ITEM_NUM_FEATURE_COLS + ITEM_NUM_INDICATOR_COLS

CAT_FEATURE_COLS = QUERY_CAT_FEATURE_COLS + ITEM_CAT_FEATURE_COLS
NUM_FEATURE_COLS = QUERY_NUM_FEATURE_COLS + ITEM_NUM_FEATURE_COLS


def returnZero():
        return 0
def returnNone():
        return None

def loadMaps():
    with open(MAPS_TO_IND_PATH, 'rb') as handle:
        maps_to_ind = pickle.load(handle)
    with open(MAPS_FROM_IND_PATH, 'rb') as handle:
        maps_from_ind = pickle.load(handle)
    return {k: defaultdict(returnZero, v) for k,v in maps_to_ind.items()}, {k: defaultdict(returnNone, v) for k,v in maps_from_ind.items()}

def saveMaps(maps_to_ind, maps_from_ind):
    maps_to_ind_dict = {k: dict(v) for k,v in maps_to_ind.items()}
    maps_from_ind_dict = {k: dict(v) for k,v in maps_from_ind.items()}

    with open(MAPS_FROM_IND_PATH, 'wb') as handle:
        pickle.dump(maps_from_ind_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(MAPS_TO_IND_PATH, 'wb') as handle:
        pickle.dump(maps_to_ind_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  

def getMappings(df_train=None, train=True, useCached = True):

    if os.path.exists(MAPS_TO_IND_PATH) and useCached:
        maps_to_ind, maps_from_ind = loadMaps()
        return maps_to_ind, maps_from_ind
    
    if train:
        maps_to_ind = {}
        maps_from_ind = {}
        for col in CAT_FEATURE_COLS:
            unique = np.sort(df_train.loc[~df_train[col].isna(), col].astype('int').unique())
            map_from_ind = {i+1:_ for i,_ in enumerate(unique)} #0 is the default for unknown id's
            map_to_ind = {_:i+1 for i,_ in enumerate(unique)}
            maps_from_ind[col] = defaultdict(returnNone,map_from_ind)
            maps_to_ind[col] = defaultdict(returnZero,map_to_ind)

        saveMaps(maps_to_ind, maps_from_ind)  
  
        return maps_to_ind, maps_from_ind
    else:
        raise Exception("Mapping cannot be loaded, first needs to be generated using train=True")


REPLACE_DICT = {'comp1_rate':{-9223372036854775808:0}}
def valueReplace(df_in, inplace=True):
    if not inplace:
        df = df_in.copy()
    else:
        df = df_in
    for col, replacements in REPLACE_DICT.items():
        for to_replace, value in replacements.items():
            df = df.replace(to_replace=to_replace, value=value)
    return df

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
        if col == 'comp1_rate':
            print(mapping)
        print(f"Applying mapping of length {len(mapping)} for {col}")
        df[col] = df[col].progress_apply(lambda val: mapping[val])
    return df
        
def getTrainData(useCached = True):
    if os.path.exists(TRAIN_SET_PATH) and useCached:
        with open(TRAIN_SET_PATH, 'rb') as handle:
            df_train = pickle.load(handle)
            print("Loaded train_data from disk")
            return df_train
    else:        
        df_train = pd.read_csv(RAW_TRAIN_DATA_PATH)
        df_train['index_srch_id'] = df_train['srch_id']
        df_train['index_prop_id'] = df_train['prop_id']
        maps_to_ind, maps_from_ind = getMappings(df_train, useCached=useCached, train=True)
        df_train = applyMapping(df_train, maps_to_ind, inplace=True)
        df_train, nanIndicColumns = addNaNIndicator(df_train)
        #df_train = df_train[CAT_FEATURE_COLS + list(df_train.columns.difference(CAT_FEATURE_COLS))] # move cat columns to the front
        with open(TRAIN_SET_PATH, 'wb') as handle:
            pickle.dump(df_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return df_train
    
def getValData(useCached = True):
    if os.path.exists(VAL_SET_PATH) and useCached:
        with open(VAL_SET_PATH, 'rb') as handle:
            df_val = pickle.load(handle)
            print("Loaded val_data from disk")
            return df_val
    else:        
        df_val = pd.read_csv(RAW_VAL_DATA_PATH)
        df_val['index_srch_id'] = df_val['srch_id']
        df_val['index_prop_id'] = df_val['prop_id']
        maps_to_ind, maps_from_ind = getMappings(df_val, useCached=True, train=False)
        df_val = applyMapping(df_val, maps_to_ind, inplace=True)
        df_val, nanIndicColumns = addNaNIndicator(df_val)
        #df_train = df_train[CAT_FEATURE_COLS + list(df_train.columns.difference(CAT_FEATURE_COLS))] # move cat columns to the front
        with open(VAL_SET_PATH, 'wb') as handle:
            pickle.dump(df_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return df_val

def getTestData(useCached = True):
    if os.path.exists(TEST_SET_PATH) and useCached:
        with open(TEST_SET_PATH, 'rb') as handle:
            df_test = pickle.load(handle)
            print("Loaded val_data from disk")
            return df_test
    else:        
        print('Loading test data')
        df_test = pd.read_csv(RAW_TEST_DATA_PATH)
        print('Loaded test data')
        df_test['index_srch_id'] = df_test['srch_id']
        df_test['index_prop_id'] = df_test['prop_id']
        maps_to_ind, maps_from_ind = getMappings(df_test, useCached=True, train=False)
        df_test = applyMapping(df_test, maps_to_ind, inplace=True)
        print('Applied mapping')
        df_test, nanIndicColumns = addNaNIndicator(df_test)
        #df_train = df_train[CAT_FEATURE_COLS + list(df_train.columns.difference(CAT_FEATURE_COLS))] # move cat columns to the front
        with open(TEST_SET_PATH, 'wb') as handle:
            pickle.dump(df_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved to disk")
        return df_test
    
def getTrainArrays(useCached =True):
    if os.path.exists(TRAIN_QUERY_ARRAY_PATH) and useCached:
        df_train_query = np.load(TRAIN_QUERY_ARRAY_PATH)
        df_train_item  = np.load(TRAIN_ITEM_ARRAY_PATH)
        df_train_rel  = np.load(TRAIN_REL_ARRAY_PATH)
        df_train_index  = np.load(TRAIN_INDEX_ARRAY_PATH)
        return df_train_query, df_train_item, df_train_rel, df_train_index
    else:
        df_train = getTrainData(useCached)
        df_train_index = df_train[['index_srch_id', 'index_prop_id']].values
        df_train_query = df_train[QUERY_FEATURE_COLS].values
        df_train_item  = df_train[ITEM_FEATURE_COLS].values
        df_train_rel = np.where(df_train['booking_bool'] == 0, df_train['click_bool'], df_train['booking_bool'])
        np.save(arr= df_train_query, file=TRAIN_QUERY_ARRAY_PATH)
        np.save(arr= df_train_item, file=TRAIN_ITEM_ARRAY_PATH)
        np.save(arr= df_train_rel, file=TRAIN_REL_ARRAY_PATH)
        np.save(arr= df_train_index, file=TRAIN_INDEX_ARRAY_PATH)
        return df_train_query, df_train_item, df_train_rel, df_train_index

def getValArrays(useCached =True):
    if os.path.exists(VAL_QUERY_ARRAY_PATH) and useCached:
        df_val_query = np.load(VAL_QUERY_ARRAY_PATH)
        df_val_item  = np.load(VAL_ITEM_ARRAY_PATH)
        df_val_rel  = np.load(VAL_REL_ARRAY_PATH)
        df_val_index  = np.load(VAL_INDEX_ARRAY_PATH)
        return df_val_query, df_val_item, df_val_rel, df_val_index
    else:
        df_val = getValData(useCached)
        df_val_index = df_val[['index_srch_id', 'index_prop_id']].values
        df_val_query = df_val[QUERY_FEATURE_COLS].values
        df_val_item  = df_val[ITEM_FEATURE_COLS].values
        df_val_rel = np.where(df_val['booking_bool'] == 0, df_val['click_bool'], df_val['booking_bool'])
        np.save(arr= df_val_query, file=VAL_QUERY_ARRAY_PATH)
        np.save(arr= df_val_item, file=VAL_ITEM_ARRAY_PATH)
        np.save(arr= df_val_rel, file=VAL_REL_ARRAY_PATH)
        np.save(arr= df_val_index, file=VAL_INDEX_ARRAY_PATH)
        return df_val_query, df_val_item, df_val_rel, df_val_index
    
def getTestArrays(useCached =True):
    if os.path.exists(TEST_QUERY_ARRAY_PATH) and useCached:
        df_test_query = np.load(TEST_QUERY_ARRAY_PATH)
        df_test_item  = np.load(TEST_ITEM_ARRAY_PATH)
        df_test_rel  = np.load(TEST_REL_ARRAY_PATH)
        df_test_rel = None
        df_test_index  = np.load(TEST_INDEX_ARRAY_PATH)
        return df_test_query, df_test_item, df_test_rel, df_test_index
    else:
        df_test = getTestData(useCached)
        df_test_index = df_test[['index_srch_id', 'index_prop_id']].values
        df_test_query = df_test[QUERY_FEATURE_COLS].values
        df_test_item  = df_test[ITEM_FEATURE_COLS].values
        df_test_rel = None
        np.save(arr= df_test_query, file=TEST_QUERY_ARRAY_PATH)
        np.save(arr= df_test_item, file=TEST_ITEM_ARRAY_PATH)
        np.save(arr= df_test_rel, file=TEST_REL_ARRAY_PATH)
        np.save(arr= df_test_index, file=TEST_INDEX_ARRAY_PATH)
        return df_test_query, df_test_item, df_test_rel, df_test_index
    
def createRawFiles():
    #Split training set in training and val, keeping srch_id together
    from sklearn.model_selection import train_test_split
    if not os.path.exists('training_set_VU_DM.csv'):
        return FileNotFoundError(f"Cannot find required file: training_set_VU_DM.csv")
    
    df = pd.read_csv('training_set_VU_DM.csv')
    all_srch_ids = df['srch_id'].unique()
    n_srch_ids = len(all_srch_ids)
    n_val_ids = int(N_VAL_RECORDS/len(df)*n_srch_ids)
    print(f"Creating train set of {n_srch_ids-n_val_ids} ids and val set with {n_val_ids} ids")
    train_srch_ids, val_srch_ids = train_test_split(all_srch_ids, test_size=n_val_ids)
    df_train = df[df['srch_id'].isin(train_srch_ids)]
    df_val = df[df['srch_id'].isin(val_srch_ids)]
    df_train.to_csv(RAW_TRAIN_DATA_PATH)
    df_val.to_csv(RAW_VAL_DATA_PATH)

class TrainDataLoader():
    def __init__(self, batch_size, negFrac, crossFrac):
        self.train_query, self.train_item, self.train_rel, self.train_index = getTrainArrays(useCached=True)
        self.batch_size = batch_size
        self.pos_ind = np.argwhere(self.train_rel>0)[:,0]
        self.neg_ind = np.argwhere(self.train_rel==0)[:,0]
        self.all_ind = list(range(self.train_item.shape[0]))
        self.nNeg = int(negFrac * batch_size)
        self.nCross = int(crossFrac * batch_size)
        self.nPos = batch_size - self.nNeg - self.nCross
        self.queryNCat = len(QUERY_CAT_FEATURE_COLS)
        self.itemNCat = len(ITEM_CAT_FEATURE_COLS)
        self.nRecords = self.train_item.shape[0]
        self.nBatches = int(self.train_item.shape[0]/self.batch_size)
        self._sampleIndices()

    def _sampleIndices(self):
        #pregenerate all sampled indices 
        self.pos_ind_sample = np.random.choice(self.pos_ind, replace=True, size=(self.nRecords))
        self.neg_ind_sample = np.random.choice(self.neg_ind, replace=True, size=(self.nRecords))
        self.cross_ind1_sample = np.random.choice(self.all_ind, replace=True, size=(self.nRecords))
        self.cross_ind2_sample = np.random.choice(self.all_ind, replace=True, size=(self.nRecords))

    def _samplePositive(self, n):
        ind_sample = self.pos_ind_sample[self.i*self.batch_size:self.i*self.batch_size+n]
        weights= self.train_rel[ind_sample]
        return ind_sample, weights

    def _sampleNegative(self, n):
        weights= np.zeros(shape=(n))-1
        ind_sample = self.neg_ind_sample[self.i*self.batch_size:self.i*self.batch_size+n]
        return ind_sample, weights


    def _crossSampleNegative(self, n):
        ind1 = self.cross_ind1_sample[self.i*self.batch_size:self.i*self.batch_size+n]
        ind2 = self.cross_ind2_sample[self.i*self.batch_size:self.i*self.batch_size+n]
        weights= np.zeros(shape=(n))-1
        return ind1, ind2, weights

    def __iter__(self):
        self.i = 0
        self._sampleIndices()
        return self

    def __next__(self):
        if self.i >= self.__len__(): raise StopIteration
        res = self.__getitem__(self.i)
        self.i += 1
        return res
    
    def __len__(self):
        return self.nBatches
    
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
         
class ValDataLoader(): #simply return batches, but one srch_id may not be spread over batches
    def __init__(self, batch_size, mode='val'):
        self.mode=mode
        if mode=='val':
            self.val_query, self.val_item, self.val_rel, self.val_index = getValArrays(useCached=True)
        if mode=='test':
            self.val_query, self.val_item, self.val_rel, self.val_index = getTestArrays(useCached=True)

        self.batch_size = batch_size
        self.queryNCat = len(QUERY_CAT_FEATURE_COLS)
        self.itemNCat = len(ITEM_CAT_FEATURE_COLS)
        self.nRecords = self.val_item.shape[0]
        self.batches = self._getBatches()
        self.nBatches = len(self.batches)

    def _getBatches(self):
        batches = []
        currentBatch = []
        i = 0
        while i <= self.nRecords:
            currentBatch.append(i)
            if i == self.nRecords-1:
                batches.append(currentBatch)
                break
            if (len(currentBatch) > self.batch_size and not self.val_index[i,0] == self.val_index[i+1,0]):
                batches.append(currentBatch)
                currentBatch = []
            i+=1
        return batches 

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.__len__(): raise StopIteration
        res = self.__getitem__(self.i)
        self.i += 1
        return res
    
    def __len__(self):
        return self.nBatches
    
    def __getitem__(self, i):
        """
        returns:
            query_cat: torch.Tensor, (batch_size, nCatQueryFeatures)
            query_num: torch.Tensor, (batch_size, nNumQueryFeatures)
            item_cat: torch.Tensor,(batch_size, nCatItemFeatures)
            item_num: torch.Tensor, (batch_size, nNumItemFeatures)
            w: torch.Tensor, (batch_size, 1)
                -1 if unrelated 1 if clicked, 5 if booked
                None if in test mode
            index: np.ndArray, array containing srch_id, prop_id index for the batch
        """
       
        ind = self.batches[i]
        X_query_cat, X_query_num, X_item_cat, X_item_num = self.val_query[ind,:self.queryNCat], self.val_query[ind,self.queryNCat:], self.val_item[ind,:self.itemNCat], self.val_item[ind,self.itemNCat:]
        X_query_cat, X_query_num, X_item_cat, X_item_num = torch.Tensor(X_query_cat).long(), torch.Tensor(X_query_num).float(), torch.Tensor(X_item_cat).long(), torch.Tensor(X_item_num).float()
        index = self.val_index[ind]
        if self.mode=='train':
            w = torch.Tensor(self.val_rel[ind]).float()
            w.requires_grad_(False)
        elif self.mode=='test':
            w = None
        return  X_query_cat, X_query_num, X_item_cat, X_item_num , w, index
            
if __name__ == "__main__":
    if not os.path.exists(RAW_TRAIN_DATA_PATH):
        print(f"Splitting into train/val")
        createRawFiles() 

    #Start sequence of data processing, all saved to disk
    print(f"Creating train arrays")
    getTrainArrays(useCached=False)

    print(f"Creating val arrays")
    getValArrays(useCached=False)

    print(f"Creating test arrays")
    getValArrays(useCached=False)

    print('Done')
        


    

    



   
