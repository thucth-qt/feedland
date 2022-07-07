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

from dataset import FeedlaneDataset

DATA_DIR = "/content/feedlane/data/classified_data"

class DeepRegression(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = models.resnet18()
        model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=1, bias=True)
            )
        self.model = model
        self.transform = None

    def forward(self, x):
        if self.transform:
            x = self.transform(x)
        if len(x) == 3:
            x = x[None]
        y = self.model(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.to(torch.float32)
        y = y.view(-1,1)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self.validation_y_hats = torch.tensor([], dtype=torch.float32).cuda()
        self.validation_preds = torch.tensor([], dtype=torch.int32).cuda()
        self.validation_targets = torch.tensor([], dtype=torch.int32).cuda()

    def on_validation_epoch_end(self) :
        # loss
        loss = F.mse_loss(self.validation_y_hats, self.validation_targets)
        acc = accuracy(self.validation_preds, self.validation_targets)
        self.log('val_loss', loss)
        self.log('val_acc', acc)


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.to(torch.float32)
        y = y.view(-1,1)
        y_hat = self.model(x)
        y_hat_round = torch.round(y_hat.clone().detach()).int()

        self.validation_y_hats = torch.cat((self.validation_y_hats, y_hat))
        self.validation_preds = torch.cat((self.validation_preds, y_hat_round))
        self.validation_targets = torch.cat((self.validation_targets, y.clone().detach().int()))


    def on_test_epoch_start(self):
        self.test_y_hats = torch.tensor([], dtype=torch.float32).cuda()
        self.test_preds = torch.tensor([], dtype=torch.int32).cuda()
        self.test_targets = torch.tensor([], dtype=torch.int32).cuda()

    def on_test_epoch_end(self) :
        # loss and acc
        loss = F.mse_loss(self.test_y_hats, self.test_targets)
        acc = accuracy(self.test_preds, self.test_targets)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        # confusion matrix
        num_classes = 4
        confmat = torchmetrics.ConfusionMatrix(num_classes).cuda()
        matrix = confmat(self.test_preds,  self.test_targets)
        plt.figure(figsize=(10,7))
        sn.heatmap(np.array(matrix.cpu()), annot=True, fmt='g') # font size
        plt.show()
        

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.to(torch.float32)
        y = y.view(-1,1)
        y_hat = self.model(x)
        y_hat_round = torch.round(y_hat.clone().detach()).int()

        self.test_y_hats = torch.cat((self.test_y_hats, y_hat))
        self.test_preds = torch.cat((self.test_preds, y_hat_round))
        self.test_targets = torch.cat((self.test_targets, y.clone().detach().int()))
        
    def set_transform(self, transform=None):
        self.transform = transform

transform = {
    'train':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    'val':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
def dataloader(phase='train'):
    assert phase in ['train', 'val']
    dataset_ = FeedlaneDataset(root_dir=DATA_DIR, transform=transform[phase], phase=phase)
    
    shuffle = True 
    if phase == 'val': shuffle = False
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
    val_loader = dataloader('val')

    trainer, model, checkpoint_callback = prepare()
    # testing
    trainer.test(model, val_loader, ckpt_path=checkpoint_callback.best_model_path)

def validate(ckpt_path="/content/lightning_logs/top_val_loss_models/epoch=29-step=4800.ckpt"):
    model = DeepRegression.load_from_checkpoint(ckpt_path)
    # model.set_transform(transforms.ToTensor())
    model.eval()

    label_dict = {
                'empty': 0.0,
                'minimal': 1.0,
                'normal': 2.0,
                'full': 3.0,
            }
    classes = label_dict.keys()
    for class_ in classes:
        # class_ = ""
        TARGETS = []
        PREDS = []
        for f in glob.glob(os.path.join("{0}/val/{1}".format(DATA_DIR,class_), "*.jpg")):
            TARGETS.append(label_dict[class_])
            # img = torch.tensor(cv2.imread(file))
            img = Image.open(f)
            img = transform['val'](img)
            # import pdb;pdb.set_trace()
            pred = model(img).detach().numpy().item()
            PREDS.append(pred)
        # import pdb;pdb.set_trace()
        loss = F.mse_loss(torch.tensor(PREDS), torch.tensor(TARGETS))
        print("{0} MSELoss: {1}".format(class_, loss))

if __name__=="__main__":
    validate("/content/feedlane/output/train/lightning_logs/lightning_logs/top_val_loss_models/epoch=92-step=13950.ckpt")