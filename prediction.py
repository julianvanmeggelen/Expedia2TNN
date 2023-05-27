import pandas as pd
import numpy as np
import os 
from collections import defaultdict
import pickle 
import torch
import argparse
from datetime import datetime

from data import ValDataLoader, TEST_INDEX_ARRAY_PATH, getTestArrays
from evaluate import valPrediction
from model import getDefaultModel

PRED_DIR = './predictions/'

def makePrediction(checkpoint_dir, p=5):
    tdl = ValDataLoader(mode='test', batch_size=50000)
    print("Loading checkpoint")
    with open(f"{os.path.dirname(checkpoint_dir)}/modelCfg.pkl", 'rb') as modelCfgFile:
        modelCfg = pickle.load(modelCfgFile)
        print("Loaded model config")
    state =torch.load(checkpoint_dir)
    mod = getDefaultModel(modelCfg=modelCfg)
    mod.load_state_dict(state)
    print("Loaded state dict")
    resultsDf = valPrediction(valDataLoader=tdl, model=mod, verbose=True)
    resultsDf = resultsDf.sort_values(by=['index_srch_id', 'sim'], ascending=False)
    resultsDf['rank'] = resultsDf.groupby(by=['index_srch_id']).cumcount()+1
    resultsDf = resultsDf.groupby('index_srch_id').head(p) #only keep p best predictions
    savedir = f"{PRED_DIR}predictions_{datetime.now()}.csv"
    resultsDf = resultsDf[['index_srch_id', 'index_prop_id']]
    resultsDf.columns = ['srch_id','prop_id']
    resultsDf.reset_index(drop=True).to_csv(savedir, index=False)
    print(f"saved predictions to {savedir}")


if __name__ == "__main__":
    if not os.path.exists('./predictions'):
        os.mkdir('./predictions')

    parser = argparse.ArgumentParser(description='Make predictions using the model')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Checkpoint directory')

    args = parser.parse_args()
    if not os.path.exists(TEST_INDEX_ARRAY_PATH):
        print("Processing test data (only happens once)")
        getTestArrays(useCached=False)
        print(f"Done, starting prediction")

    makePrediction(checkpoint_dir=args.checkpoint_dir)
    