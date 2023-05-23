import ml_collections as mlc
from model import getDefaultModel, weightedCoSim
from data import TrainDataLoader
from config import trainCfg as tCfg
import torch.optim as opt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
writer = SummaryWriter()
import os
from datetime import datetime


def train(state_init:str=None):
    logDirName = f"{datetime.now()}"
    os.mkdir("./training_log/"+logDirName)
    mod = getDefaultModel()
    if state_init:
        state =torch.load(state_init)
        mod.load_state_dict(state)
    dl = TrainDataLoader(batch_size=tCfg.batch_size, negFrac=tCfg.negFrac, crossFrac=tCfg.crossFrac)
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
            #print(f"Batch {b} - {loss.item():.5f}")
            writer.add_scalar('Loss/train/batch', loss.item(), nBatches*epoch +b)
            epochLoss+= loss.item()
        writer.add_scalar('Loss/train/epoch', epochLoss/nBatches, epoch)
        print(f"Train loss: {epochLoss/nBatches}")
        epochLoss = 0.0
        torch.save(mod.state_dict(),"./training_log/"+logDirName+f"/state_epoch_{epoch}")

        


if __name__ == "__main__":
    train()
    

