import sys, os, glob

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from datetime import datetime
from PIL import Image

from dataset import FeedlaneDataset, data_transform
from model import DeepRegression

DATA_DIR = "/content/feedlane/data/classified_data"
OUTPUT_DIR = "/content/feedlane/output"
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def dataloader(phase='train'):
    assert phase in ['train', 'val', 'test']
    dataset_ = FeedlaneDataset(root_dir=DATA_DIR, transform=data_transform[phase], phase=phase)
    
    shuffle = True 
    if phase != 'train': shuffle = False
    # Dataloader
    return DataLoader(dataset_, batch_size=32, shuffle=shuffle)

def prepare():
    # model
    model = DeepRegression()

    ckpt_dir = "/content/feedlane/output/train/lightning_logs/top_val_loss_models"
    os.makedirs(ckpt_dir, exist_ok=True)
    # checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir, save_top_k=2, monitor="val_loss")

    # trainer
    trainer = pl.Trainer(max_epochs=200, check_val_every_n_epoch=3, devices=1, accelerator="gpu", log_every_n_steps=3, callbacks=[checkpoint_callback])

    return trainer, model, checkpoint_callback

def train():
    train_loader = dataloader('train')
    val_loader = dataloader('val')

    trainer, model, _ = prepare()
    # training
    trainer.fit(model, train_loader, val_loader)

def test():
    val_loader = dataloader('test')

    trainer, model, checkpoint_callback = prepare()
    # testing
    trainer.test(model, val_loader, ckpt_path=checkpoint_callback.best_model_path)

if __name__=="__main__":
    train()