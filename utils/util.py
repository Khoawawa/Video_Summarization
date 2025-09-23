import numpy as np
from torch.utils.data import Dataset, Dataloader

class CustomDataset(Dataset):
    def __init__(self, inputs):
        self.content = inputs
    def __len__(self):
        return len(self.content)
    def __getitem__(self, idx):
        return self.content[idx]
    
