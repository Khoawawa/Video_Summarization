import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import os
import cv2

csv_file = '../Video_Summarization/preprocessing/video_and_keyframe_path.csv'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = pd.read_csv(df)
        self.transform = transform
        self.img_dir = img_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.img_dir, row['image_name'])
        caption = row['comment']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption
def load_image_loaders(csv_path, args, mode='train'):
    global transform
    dataset = ImageDataset(csv_path,args.img_dir, transform)
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
    for phase in phases:
        if phase == 'train':
            loaders[phase] = DataLoader(train_dataset, batch_size=args.data_config['batch_size'], shuffle=True)
        elif phase == 'val':
            loaders[phase] = DataLoader(val_dataset, batch_size=args.data_config['batch_size'], shuffle=False)
        else:
            loaders[phase] = DataLoader(test_dataset, batch_size=args.data_config['batch_size'], shuffle=False)
    return loaders
    
class KeyframeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if isinstance(csv_file, str):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = csv_file.copy()
        self.transform = transform

        # Parse nếu column là string
        if isinstance(self.df['keyframe_paths'].iloc[0], str):
            self.df['keyframe_paths'] = self.df['keyframe_paths'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']
        key_frame_paths = row['keyframe_paths']
        caption = row['captions']
        images = []
        if len(key_frame_paths) > 0:
            for img_path in key_frame_paths:
                try:
                    root_dir = os.path.dirname(os.path.abspath(csv_file))
                    img_path = os.path.join(root_dir, img_path)
                    img_path = os.path.normpath(img_path)
                    img = Image.open(img_path)
                except:
                    # nếu lỗi thì tạo ảnh trắng
                    img = Image.fromarray((0 * np.ones((224,224), dtype=np.uint8)))
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        else:
            # fallback: không có frame → 1 ảnh trắng
            img = Image.fromarray((255 * np.ones((224,224), dtype=np.uint8)))
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack thành tensor [num_frames, C, H, W]
        images = torch.stack(images, dim=0)

        return {
            "video_id": str(video_id),
            "caption": caption,
            "images": images,
            "num_key_frames": row['num_key_frames']
        }

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize về 224x224
    transforms.ToTensor(),           # đổi sang Tensor [0,1]
    transforms.Normalize([0.5], [0.5])   # chuẩn hóa std
])

def prepare_csv_with_split(csv_file):
    df = pd.read_csv(csv_file)

    # chia ngẫu nhiên train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)  # 70/20/10

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    df = pd.concat([train_df, val_df, test_df])
    df.to_csv(csv_file, index=False)
    return df

def collate_fn(batch):
    video_ids = [item["video_id"] for item in batch]
    images = [item["images"] for item in batch]  # list of [num_frames, C, H, W]
    num_key_frames = [item["num_key_frames"] for item in batch]
    caption = [item['caption'] for item in batch]
    return {
        "video_id": video_ids,
        "caption": caption,
        "images": images,  # giữ list vì num_frames khác nhau
        "num_key_frames": torch.tensor(num_key_frames)
    }

def load_datadict(csv_file, batch_size=8, mode="train"):
    df = pd.read_csv(csv_file)

    if 'split' not in df.columns:
        df = prepare_csv_with_split(csv_file)

    loaders = {}
    phases = ['train', 'val', 'test'] if mode != "test" else ['test']

    for phase in phases:
        df_phase = df[df['split'] == phase]
        '''temp_csv = f"..Video_Summarization/utils/data/{phase}.csv"
        df_phase.to_csv(temp_csv, index=False)'''

        dataset = KeyframeDataset(df_phase, transform=transform)
        loaders[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=(phase == 'train'),
                                    collate_fn=collate_fn)  # để tránh lỗi do số frame khác nhau

    return loaders

class VideoTensorDataset(Dataset):
    def __init__(self, df, csv_file:str, transform=None, num_frames=16):
        self.df = df.copy()
        self.csv_file = csv_file

        self.transform = transform
        self.num_frames = num_frames


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global csv_file
        row = self.df.iloc[idx]
        video_id = row['video_id']
        video_path = row['video_path']
        caption = row['captions']
        root_dir = os.path.dirname(os.path.abspath(self.csv_file))
        video_path = os.path.join(root_dir, video_path)
        video_path = os.path.normpath(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # fallback: trả tensor trắng nếu không mở được
            dummy = torch.zeros((self.num_frames, 224, 224))
            return {"video_id": str(video_id), "video_tensor": dummy, "num_frames": 0}

        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Chọn frame ngẫu nhiên hoặc đều cách (ở đây chia đều)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (224, 224))
                img = transforms.ToTensor()(frame)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
        cap.release()

        if len(frames) == 0:
            frames = [torch.zeros(224, 224) for _ in range(self.num_frames)]

        # Stack thành tensor [num_frames, C, H, W]
        video_tensor = torch.stack(frames, dim=0)
        return {
            "video_id": str(video_id),
            "captions": caption,
            "video_tensor": video_tensor,
            "num_frames": len(frames)
        }

def collate_video_fn(batch):
    video_ids = [item["video_id"] for item in batch]
    video_tensors = [item["video_tensor"] for item in batch]
    num_frames = [item["num_frames"] for item in batch]
    caption = [item["captions"] for item in batch]
    return {
        "video_id": video_ids,
        "captions": caption,
        "video_tensor": video_tensors,  # list do num_frames có thể khác nhau
        "num_frames": torch.tensor(num_frames)
    }


def load_video_loaders(csv_file, batch_size=2, mode="train", num_frames=16):
    df = pd.read_csv(csv_file)
    if 'split' not in df.columns:
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        df = pd.concat([train_df, val_df, test_df])
        df.to_csv(csv_file, index=False)

    loaders = {}
    phases = ['train', 'val', 'test'] if mode != "test" else ['test']
    for phase in phases:
        df_phase = df[df['split'] == phase]
        dataset = VideoTensorDataset(df_phase,csv_file=csv_file, num_frames=num_frames)
        loaders[phase] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(phase == "train"),
            collate_fn=collate_video_fn
        )
    return loaders


if __name__ == '__main__':
    csv_file = 'preprocessing/video_and_keyframe_path.csv'
    loader = load_video_loaders(csv_file)
    print(next(iter(loader['train'])))