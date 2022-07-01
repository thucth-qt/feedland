import os
import math

import numpy as np
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader


class FeedLaneDataset():
    def __init__(self) -> None:
        # Create PyTorch data generators
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
        label_dict_raw = {
            "empty": 0,
            "minimal": 1,
            "normal": 2,
            "full": 3,
        }
        label_dict = {
            0:0,
            1:3,
            2:1,
            3:2,
        }
        data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ]),
            'val':
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                self.normalize
            ]),
        }

        self.image_datasets = {
            'train': 
            datasets.ImageFolder('data/classified_data/train', data_transforms['train'], target_transform=lambda x: label_dict[x]),
            'val': 
            datasets.ImageFolder('data/classified_data/val', data_transforms['val'], target_transform=lambda x: label_dict[x])
        }
    
    def get_len(self, data_type):
        return len(self.image_datasets[data_type])

    def data_loader(self):
        dataloaders = {
        'train':DataLoader(self.image_datasets['train'],
                                    batch_size=32,
                                    shuffle=True, num_workers=4),
        'val':DataLoader(self.image_datasets['val'],
                                    batch_size=8,
                                    shuffle=False, num_workers=4)
        }

        return dataloaders