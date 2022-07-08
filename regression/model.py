import os
import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from dataset import FeedLaneDataset

class FeedLaneModel():
    def __init__(self):
        # Create the network
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.model = models.resnet18(pretrained=True).to(self.device)
            
        # for param in self.model.parameters():
        #     param.requires_grad = False   

        self.model.fc = nn.Sequential(
                    # nn.Linear(512, 128),
                    # nn.ReLU(inplace=True),
                    nn.Linear(512, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

        os.makedirs("savemodel/pytorch", exist_ok=True)
        self.ckpt_path = 'savemodel/pytorch/weights.pth'

    def count_correct(self, preds, labels, epsilon):
        running_corrects = 0
        for idx, value in enumerate(labels):
            if value == 0.0 and preds[idx] < value + epsilon:
                running_corrects += 1
            elif value == 0.1 and preds[idx] > value - epsilon and preds[idx] < value + epsilon:
                running_corrects += 1
            elif value == 0.2 and preds[idx] > value - epsilon and preds[idx] < value + epsilon:
                running_corrects += 1
            elif value == 0.3 and preds[idx] > value - epsilon :
                running_corrects += 1
        return running_corrects

    # Train the model
    def train_model(self, dataset, num_epochs=100, epsilon=0.05):
        best_loss = 1000
        dataloader = dataset.data_loader()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 100)

            for phase in ['train']:
                if phase == 'train':
                    self.model.train()
                # else:
                #     self.model.eval()

                running_outputs = torch.tensor([])
                running_labels = torch.tensor([])

                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.type(torch.float32) / 10
                    labels = labels.to(self.device)

                    outputs = torch.squeeze(self.model(inputs))              

                    if phase == 'train':
                        loss = self.criterion(outputs, labels)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    running_outputs = torch.cat((running_outputs,outputs.cpu()))
                    running_labels = torch.cat((running_labels,labels.cpu()))


                epoch_loss = self.criterion(running_outputs, running_labels)
                epoch_acc = self.count_correct(running_outputs, running_labels, epsilon) / len(running_outputs)
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            if epoch_loss < best_loss: 
                best_loss = epoch_loss
                # torch.save(self.model, 'best-model.pt')
                print(f"Saving best model at {self.ckpt_path} with loss {best_loss}")
                torch.save(self.model.state_dict(), self.ckpt_path)
               
        return self.model

    def train(self, dataset, num_epochs):
        model_trained = self.train_model(dataset, num_epochs=num_epochs)
        
    
    def validate(self, dataset, epsilon=0.05):
        self.model.load_state_dict(torch.load(self.ckpt_path))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        val_loader = dataset.data_loader()["val"]

        running_outputs = torch.tensor([])
        running_labels = torch.tensor([])

        for inputs, labels in val_loader:
            inputs = inputs.to(self.device)
            labels = labels.type(torch.float32) / 10
            labels = labels.to(self.device)

            outputs = torch.squeeze(self.model(inputs))

            running_outputs = torch.cat((running_outputs,outputs.cpu()))
            running_labels = torch.cat((running_labels,labels.cpu()))

        loss = self.criterion(running_outputs, running_labels)
        acc = self.count_correct(running_outputs, running_labels, epsilon) / len(running_outputs)

        print('Validation loss: {:.4f}, acc: {:.4f}'.format(loss, acc))

        # confusion matrix
        num_classes = 4
        cm = confusion_matrix((running_labels*10).type(torch.int32), torch.round(running_outputs*10), labels=[i for i in range(num_classes)])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('/content/feedlane/savemodel/pytorch/cm_reg.png')

    def inference(self, file_path):
        self.model.load_state_dict(torch.load(self.ckpt_path))
        from torchvision import datasets, models, transforms

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        trans = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ]),

        img = cv2.imread(file_path)
        img = trans(img)
        # import pdb;pdb.set_trace()
        inputs = torch.unsqueeze(img, 0)

        inputs = inputs.to(self.device)

        outputs = torch.squeeze(self.model(inputs))
        print(outputs)
    

if __name__=="__main__":
    # dataset = FeedLaneDataset()
    # model = FeedLaneModel()
    # model.train(dataset, 100)
    # model.validate(dataset)
    import shutil

    ls_file = os.listdir("/Users/conglinh/document/FreeLance/feedlane/data/labelstudio")
    for f in ls_file:
        if '.jpg' in f:
            try:
                os.remove(os.path.join("/Users/conglinh/document/FreeLance/feedlane/data/data_Thuc/all_cam", f))
            except:
                pass

    # for (root, dirs, file) in os.walk():
    #     print(root)
    #     for f in file:
    #         if '.jpg' in f:
    #             input = os.path.join(root, f)
    #             shutil.move(input, "/Users/conglinh/document/FreeLance/feedlane/data/data_Thuc/all_cam")
    