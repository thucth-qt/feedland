import os
import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from .dataset import FeedLaneDataset

class FeedLaneModel():
    def __init__(self):
        # Create the network
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=False).to(self.device)
            
        for param in self.model.parameters():
            param.requires_grad = False   

        self.model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.fc.parameters())
  
    # Train the model
    def train_model(self, dataset, num_epochs=100):
        dataloader = dataset.data_loader()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 50)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.type(torch.float32)
                    labels = labels.to(self.device)

                    outputs = torch.squeeze(self.model(inputs))
                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    preds = outputs
                    running_loss += loss.detach()
                    running_corrects += torch.sum(torch.round(preds) == labels.data)
                    # if phase == 'val':
                    #     print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,running_loss.item(),running_corrects.item()))


                epoch_loss = running_loss / math.ceil(len(dataset.get_len(phase)) / 32)
                epoch_acc = running_corrects.float() / len(dataset.get_len(phase))
                
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,epoch_loss.item(),epoch_acc.item()))
        return self.model

    def train(self, dataset, num_epochs):
        model_trained = self.train_model(dataset, num_epochs=num_epochs)
        #Save model
        torch.save(model_trained.state_dict(), 'savemodel/pytorch/weights.pth')

    def validate(self, dataset):
        self.model.load_state_dict(torch.load('savemodel/pytorch/weights.pth'))
        validation_img_paths = "/content/feedlane/data/classified_data/val"
        # ls_file = []
        # for f in os.listdir(validation_img_paths)[:20]:
        #     ls_file.append(os.path.join(validation_img_paths,f))
        # for i in range(0, len(ls_file), 10):
        #     img_list = [Image.open(img_path) for img_path in ls_file[i:i+10]]
        #     validation_batch = torch.stack([data_transforms['val'](img).to(device)
        #                                 for img in img_list])
        #     pred_logits_tensor = model(validation_batch)
        #     print(pred_logits_tensor)

        self.model.eval()
        val_loader = dataset.data_loader("val")

        running_loss = 0
        running_corrects = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(self.device)
            labels = labels.type(torch.float32)
            # labels = torch.tensor(labels.clone().detach(), dtype=torch.float32)
            labels = labels.to(self.device)
            # print(labels.data)

            outputs = torch.squeeze(self.model(inputs))
            loss = self.criterion(outputs, labels)
            preds = outputs
            # import pdb; pdb.set_trace()
            running_loss += loss.detach()
            running_corrects += torch.sum(torch.round(preds) == labels.data)

        epoch_loss = running_loss / math.ceil(len(val_loader) / 32)
        epoch_acc = running_corrects.float() / len(val_loader)

        print('Validation loss: {:.4f}, acc: {:.4f}'.format(epoch_loss.item(), epoch_acc.item()))
    

if __name__=="__main__":
    dataset = FeedLaneDataset()
    model = FeedLaneModel(dataset, 100)
    model.train()