import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import pytorch_lightning as pl
import os
import glob
import cv2
import numpy as np
from torchmetrics.functional import accuracy
import torchmetrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import random
import shutil
from PIL import Image

class FeedlaneDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase="train"):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.label_dict = {
            'empty': 0.0,
            'minimal': 1.0,
            'normal': 2.0,
            'full': 3.0,
        }
        self.img_paths = []
        self.img_labels = []
        self.img_extentions = ["*.jpg", "*.bmp", "*.png"]
        self.dataset_ratio = np.array([0.8, 0.2, 0]) # (train, val, test) ratio
        self.load_dataset()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img = cv2.imread(self.img_paths[idx])
        img = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]

        if self.transform:
            img = self.transform(img)
        return img, label

    def load_dataset(self):
        for cls in os.listdir(os.path.join(self.root_dir, self.phase)):
            if cls in self.label_dict.keys():
                files = []
                for extention in self.img_extentions:
                    files.extend(glob.glob(os.path.join(self.root_dir, self.phase,cls, extention)))
                # if self.phase == "train":
                #     if cls == "full":
                #         files = files * 3
                #     elif cls == "normal":
                #         files = files * 2
                self.img_paths.extend(files)
                self.img_labels.extend([self.label_dict[cls]]*len(files))
                print(f"Class {cls} has {len(files)} file(s)")
        print(f"{self.phase} set have total {len(self.img_labels)} image(s) loaded")

    def get_len_train_val_test(self):
        n_train, n_val = (len(self)*self.dataset_ratio[:2]).astype(np.int32)
        n_test = len(self) - n_train - n_val
        return [n_train, n_val, n_test]