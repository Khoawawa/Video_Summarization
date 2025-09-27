import numpy as np
from torch.utils.data import Dataset, Dataloader

def create_model(args):
    # create model based on args
    pass
def create_loss_fn(args):
    # create loss function based on args
    pass
def load_datadict(args):
    # load data based on args
    data = np.load(args.data_path, allow_pickle=True)
    dataset = CustomDataset(data)
    dataloader = Dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
class CustomDataset(Dataset):
    def __init__(self, inputs):
        self.content = inputs
    def __len__(self):
        return len(self.content)
    def __getitem__(self, idx):
        return self.content[idx]
    
