import ml_collections as mlc
from model import getDefaultModel, weightedCoSim
from data import TrainDataLoader, ValDataLoader
from evaluate import valDcg, randomOrderingBenchmark, perfectOrderingBenchmark, valLoss
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
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def train(checkpoint_dir:str=None):

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

    logDirName = f"{datetime.now()}"
    logPath ="./training_log/"+logDirName
    os.mkdir(logPath)

    with open(f"{logPath}/modelCfg.pkl", 'wb') as file: #save config used to logdir
        pickle.dump(modelCfg, file)
        print("Saved modelCfg to log")

    mod.to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(f"Number of trainable model parameters: {get_trainable_params(mod)}")

    dl = TrainDataLoader(batch_size=tCfg.batch_size, negFrac=tCfg.negFrac, crossFrac=tCfg.crossFrac, device=DEVICE)
    vdl = ValDataLoader(batch_size=10000, device=DEVICE)
    random_dcg = randomOrderingBenchmark(dataLoader=vdl, device=DEVICE)
    print(f"Random ordering bechmark:{random_dcg}")
    perfect_dcg = perfectOrderingBenchmark(dataLoader=vdl, device=DEVICE)
    print(f"Perfect ordering bechmark:{perfect_dcg}")
    untrained_dcg = valDcg(dataLoader=vdl, model=mod)
    print(f"Untrained model DCG: {untrained_dcg}")

    optimizer = opt.Adam(params=mod.parameters(), lr = tCfg.lr)

    nBatches = len(dl)
    epochLoss = 0.0
    bestValDcg = 0.0
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
        val_loss = valLoss(vdl,mod)
        val_dcg = valDcg(model=mod, dataLoader= vdl)
        writer.add_scalar('dcg/val/epoch', val_dcg, epoch)
        writer.add_scalar('Loss/val/epoch', val_loss, epoch)
        print(f"Train loss: {epochLoss/nBatches}, val loss: {val_loss} val DCG: {val_dcg}")
        epochLoss = 0.0
        if val_dcg > bestValDcg:
            print(f"New best DCG@5 on val set ({val_dcg}>{bestValDcg}), saving checkpoint")
            torch.save(mod.state_dict(),logPath+f"/state_cp.pt")
            bestValDcg = val_dcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Checkpoint directory')

    args = parser.parse_args()
    train(checkpoint_dir=args.checkpoint_dir)
    

