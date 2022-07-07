import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
import pytorch_lightning as pl
import numpy as np
from torchmetrics.functional import accuracy
import torchmetrics
import seaborn as sn
import matplotlib.pyplot as plt

DATA_DIR = "/content/feedlane/data/classified_data"
OUTPUT_DIR = "/content/feedlane/output"
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

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


