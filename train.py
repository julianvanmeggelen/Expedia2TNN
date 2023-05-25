import ml_collections as mlc
from model import getDefaultModel, weightedCoSim
from data import TrainDataLoader, ValDataLoader
from evaluate import valDcg
from config import trainCfg as tCfg
import torch.optim as opt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
writer = SummaryWriter()
import os
from datetime import datetime
import argparse
import pickle
from config import modelCfg as defaultModelCfg


def train(checkpoint_dir:str=None):
    logDirName = f"{datetime.now()}"
    logPath ="./training_log/"+logDirName
    os.mkdir(logPath)

    if checkpoint_dir: #load checkpoint
        print("Loading checkpoint")
        with open(f"{os.path.dirname(checkpoint_dir)}/modelCfg.pkl", 'rb') as modelCfgFile:
            modelCfg = pickle.load(modelCfgFile)
            print("Loaded model config")
        state =torch.load(checkpoint_dir)
        mod = getDefaultModel(modelCfg=modelCfg)
        mod.load_state_dict(state)
        print("Loaded state dict")
    else: #start anew
        modelCfg = defaultModelCfg
        mod = getDefaultModel(modelCfg=modelCfg)
        print("Loaded  model with defaultcfg")

    with open(f"{logPath}/modelCfg.pkl", 'wb') as file: #save config used to logdir
        pickle.dump(modelCfg, file)
        print("Saved modelCfg to log")

    dl = TrainDataLoader(batch_size=tCfg.batch_size, negFrac=tCfg.negFrac, crossFrac=tCfg.crossFrac)
    vdl = ValDataLoader(batch_size=10000)

    optimizer = opt.Adam(params=mod.parameters(), lr = tCfg.lr)

    nBatches = len(dl)
    epochLoss = 0.0
    print(f"Starting training for {tCfg.n_epoch} epochs with {nBatches} batches.")
    for epoch in range(tCfg.n_epoch):
        for  b, (X_query_cat, X_query_num, X_item_cat, X_item_num , w) in tqdm(enumerate(dl), total=dl.nBatches):
            e_q, e_i = mod(X_query_cat, X_query_num, X_item_cat, X_item_num)
            loss = weightedCoSim(w,e_q,e_i)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #print(f"Batch {b} - {loss.item():.5f}")
            writer.add_scalar('Loss/train/batch', loss.item(), nBatches*epoch +b)
            epochLoss+= loss.item()
        writer.add_scalar('Loss/train/epoch', epochLoss/nBatches, epoch)
        val_dcg = valDcg(model=mod, dataLoader= vdl)
        writer.add_scalar('dcg/val/epoch', val_dcg, epoch)
        print(f"Train loss: {epochLoss/nBatches}, val DCG: {val_dcg}")
        epochLoss = 0.0
        torch.save(mod.state_dict(),logPath+f"/state_epoch_{epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Checkpoint directory')

    args = parser.parse_args()
    train(checkpoint_dir=args.checkpoint_dir)
    

