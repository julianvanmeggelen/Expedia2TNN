import ml_collections as mlc
from model import getDefaultModel, weightedCoSim
from data import TrainDataLoader
from config import trainCfg as tCfg
import torch.optim as opt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def train():
    mod = getDefaultModel()
    dl = TrainDataLoader(batch_size=tCfg.batch_size, negFrac=tCfg.negFrac, crossFrac=tCfg.crossFrac)
    optimizer = opt.Adam(params=mod.parameters(), lr = tCfg.lr)
    loss = 0.0
    print(f"Starting training for {tCfg.n_epochs} epochs with {len(dl)} batches.")
    for epoch in range(tCfg.n_epoch):
        for  b, (X_query_cat, X_query_num, X_item_cat, X_item_num , w) in enumerate(dl):
            e_q, e_i = mod(X_query_cat, X_query_num, X_item_cat, X_item_num)
            loss = weightedCoSim(w,e_q,e_i)
            loss.backward()
            optimizer.step()
            print(f"Batch {b} - {loss.item():.5f}")
            writer.add_scalar('Loss/train', loss.item())


if __name__ == "__main__":
    train()
    

