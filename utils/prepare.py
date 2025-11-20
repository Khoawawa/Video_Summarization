import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import os
import cv2

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class ImageDataset(Dataset):
    def __init__(self, csv_path, img_dir,prefix_len:int, tokenizer_type='gpt2', transform=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_type)
        self.prefix_len = prefix_len
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.transform = transform
        self.max_seq_len = 33 # 99% of captions are less than 33 (check untitle7.ipynb)
        self.img_dir = img_dir
        #caption goes here
        self.caption_tokens = [
            torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
            for caption in self.df['comment']
        ]
        
    def pad_token(self, caption_tokens):
        tokens = caption_tokens
        padding = self.max_seq_len - caption_tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.long) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)
        tokens[~mask] = 0 # mask out padding
        # creating the attention mask including the prefix
        mask = torch.cat([torch.ones(self.prefix_len, dtype=torch.long), mask], dim=0)
        return tokens, mask
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(image_path).convert('RGB')
        # get caption and pad
        tokens = self.caption_tokens[index]
        tokens, mask = self.pad_token(tokens)
        if self.transform:
            image = self.transform(image)
        return image, tokens, mask
def load_image_loaders(csv_path, args, prefix_len:int,mode='train'):
    global transform
    true_csv_path = os.path.join(args.data_config['data_dir'], csv_path)
    true_img_dir = os.path.join(args.data_config['data_dir'], args.data_config['img_dir'])
    dataset = ImageDataset(csv_path=true_csv_path, img_dir=true_img_dir,prefix_len=prefix_len, transform=transform)
    def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        total = len(dataset)
        train_size = int(train_ratio * total)
        val_size = int(val_ratio * total)
        test_size = total - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size],generator=torch.Generator().manual_seed(seed))
        
        return train_dataset, val_dataset, test_dataset
    loaders = {}
    phases = ['train', 'val', 'test'] if mode != "test" else ['test']
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    pin_memory = torch.cuda.is_available()
    for phase in phases:
        shuffle = (phase == 'train')
        if phase == 'train':
            data = train_dataset
        elif phase == 'val':
            data = val_dataset
        else:
            data = test_dataset
        
        loaders[phase] = DataLoader(
            data,
            batch_size=args.data_config['batch_size'],
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
    return loaders

if __name__ == '__main__':
    pass