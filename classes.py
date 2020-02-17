import os
import zipfile
import gc

import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN # pytorch mtcnn works poor

import albumentations as albu
from albumentations.pytorch import ToTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TinkoffDataset(Dataset):
    """Tinkoff dataset."""

    def __init__(self, csv_file, mtcnn_model, root_dir, img_ids, sh=15, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.img_ids = img_ids
        self.transform = transform
        self.mtcnn_model = mtcnn_model
        self.sh = sh # not only face but head

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        image_name = os.path.join(self.root_dir,
                                self.df.iloc[self.img_ids[idx], 0])

        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
        label = self.df.iloc[self.img_ids[idx], 1].astype(np.float32)
        
        result = self.mtcnn_model.detect_faces(image)

        if result:
            bb = np.array(result[0]['box']).clip(min=0) # bounding boxes
            image = image[max(0, bb[1]-self.sh):bb[1]+bb[3]+self.sh, 
                          max(0, bb[0]-self.sh):bb[0]+bb[2]+self.sh, :] # cropped face (most confident)
               
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class FNInceptionResnetV1(torch.nn.Module):
    def __init__(self):
        super(FNInceptionResnetV1, self).__init__()
        self.inner = InceptionResnetV1(pretrained='vggface2')
        self.fn_logits = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
    
    def forward(self, x):
        x = self.inner(x)
        x = self.fn_logits(x)
        return x.view(-1)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
    
def train_val_transfroms():
    train_transform = [
        albu.Resize(160, 160),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(rotate_limit=20, p=0.7),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        ToTensor(),
    ]
    
    val_transform = [
        albu.Resize(160, 160),
        ToTensor()
    ]
    return albu.Compose(train_transform), albu.Compose(val_transform)