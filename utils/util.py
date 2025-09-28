import numpy as np
from torch.utils.data import Dataset, Dataloader
import os
import torch
import json
from models.Captioner import Captioner
def create_model(args):
    # create model based on args
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath, "r") as f:
        model_config = json.load(f)[args.model]
    if "captioner" in args.model.lower():
        model = Captioner(**model_config)
    
    return model
    
def load_datadict(args):
    # load data based on args
    data = np.load(args.data_path, allow_pickle=True)
    dataset = CustomDataset(data)
    dataloader = Dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
def save_model(model_path: str, **save_dict):
    os.makedirs(os.path.split(model_path)[0], exist_ok=True)
    torch.save(save_dict, model_path)
    
class CustomDataset(Dataset):
    def __init__(self, inputs):
        self.content = inputs
    def __len__(self):
        return len(self.content)
    def __getitem__(self, idx):
        return self.content[idx]
    
