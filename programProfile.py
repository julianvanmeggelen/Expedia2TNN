"""
cProfile progam analysis
"""

import cProfile
from datetime import datetime
import argparse
from tqdm import tqdm
from data import TrainDataLoader



def main():
    dl = TrainDataLoader(2048*16, 0.3,0.2)
    i = 1
    for X_query_cat, X_query_num, X_item_cat, X_item_num , w in dl:
        i+=1
        print(X_query_cat.shape, X_query_num.shape, X_item_cat.shape, X_item_num.shape , w.shape)
        print(i)

def printProfile(f):
    import pstats
    p = pstats.Stats('./profile/' + f)
    p.sort_stats('tottime').print_stats()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", help="show profile output for file")

    args = parser.parse_args()

    filename = None

    if args.f:
        if args.f == '-1':
            import os
            print("yes")
            filenames  = os.listdir('./profile/')
            filename = sorted(filenames)[-1]
        else:
            filename = args.f
        printProfile(filename)

       

    else:
        now = datetime.now() # current date and time
        filename = f"profile_{now.strftime('%m%d%Y_%H%M%S')}"
        filedir = './profile/' + filename 
        cProfile.run('main()', filename=filedir)
        printProfile(filename)