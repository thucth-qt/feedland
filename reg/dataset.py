import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import glob
from PIL import Image

from config import FeedlaneConfig

data_transform = {
    'train':
        transforms.Compose([
            transforms.Resize(FeedlaneConfig.IMG_SIZE),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FeedlaneConfig.IMG_MEAN, FeedlaneConfig.IMG_STD)
        ]),
    'val':
        transforms.Compose([
            transforms.Resize(FeedlaneConfig.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(FeedlaneConfig.IMG_MEAN, FeedlaneConfig.IMG_STD)
        ]),
    'test':
        transforms.Compose([
            transforms.Resize(FeedlaneConfig.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(FeedlaneConfig.IMG_MEAN, FeedlaneConfig.IMG_STD)
        ])
    }

class FeedlaneDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase="train"):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
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
            if cls in FeedlaneConfig.LABEL_DICT.keys():
                files = []
                for extention in FeedlaneConfig.IMG_EXTENSIONS:
                    files.extend(glob.glob(os.path.join(self.root_dir, self.phase,cls, extention)))
                self.img_paths.extend(files)
                self.img_labels.extend([FeedlaneConfig.LABEL_DICT[cls]]*len(files))
                print(f"Class {cls} has {len(files)} file(s)")
        print(f"{self.phase} set have total {len(self.img_labels)} image(s) loaded")
