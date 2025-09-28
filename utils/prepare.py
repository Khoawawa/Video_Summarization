import torch
from torch.utils.data import Dataset, Dataloader
def load_datadict(args):
    # load data based on args
    data = {}
    loader = {}
    #TODO: chia loader thành 3 phần: train, val, test
    if args.mode == "test":
        phases = ['test']
    else:
        phases = ['train', 'val', 'test']
    for phase in phases:
        # TODO: load data tu csv vao data[phase] va dua vao loader
        data[phase] = ...
        loader[phase] = ...
    
    return loader.copy()