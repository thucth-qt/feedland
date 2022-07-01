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

# Create PyTorch data generators
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': 
    datasets.ImageFolder('data/classified_data/train', data_transforms['train']),
    'val': 
    datasets.ImageFolder('data/classified_data/val', data_transforms['val'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True, num_workers=4),
    'val':
    torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size=32,
                                shuffle=False, num_workers=4)
}

# Create the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
# model.fc = nn.Sequential(
#                nn.Linear(2048, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, 3)).to(device)
# criterion = nn.CrossEntropyLoss()

model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 64),
               nn.ReLU(inplace=True),
               nn.Linear(64, 1)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.fc.parameters())

# Train the model
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.type(torch.float32)
                # labels = torch.tensor(labels.clone().detach(), dtype=torch.float32)
                labels = labels.to(device)
                # print(labels.data)

                outputs = torch.squeeze(model(inputs))
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # _, preds = torch.max(outputs, 1)
                preds = outputs
                # print("pred: ", preds)
                # print("label: ", labels.data)
                # import pdb; pdb.set_trace()
                running_loss += loss.detach()
                running_corrects += torch.sum(torch.round(preds) == labels.data)
                # if phase == 'val':
                #     print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,running_loss.item(),running_corrects.item()))


            epoch_loss = running_loss / math.ceil(len(image_datasets[phase]) / 32)
            epoch_acc = running_corrects.float() / len(image_datasets[phase])
            
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,epoch_loss.item(),epoch_acc.item()))
    return model

def train():
    model_trained = train_model(model, criterion, optimizer, num_epochs=20)
    #Save model
    torch.save(model_trained.state_dict(), 'savemodel/pytorch/weights.pth')

def validate():
    # Inference
    model = models.resnet50(pretrained=True).to(device)
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)).to(device)
    model.load_state_dict(torch.load('savemodel/pytorch/weights.pth'))
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

    model.eval()
    val_loader = torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size=32,
                                shuffle=False, num_workers=4)

    running_loss = 0
    running_corrects = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.type(torch.float32)
        # labels = torch.tensor(labels.clone().detach(), dtype=torch.float32)
        labels = labels.to(device)
        # print(labels.data)

        outputs = torch.squeeze(model(inputs))
        loss = criterion(outputs, labels)
        preds = outputs
        # import pdb; pdb.set_trace()
        running_loss += loss.detach()
        running_corrects += torch.sum(torch.round(preds) == labels.data)

    epoch_loss = running_loss / math.ceil(len(val_loader) / 32)
    epoch_acc = running_corrects.float() / len(val_loader)

    print('Validation loss: {:.4f}, acc: {:.4f}'.format(epoch_loss.item(), epoch_acc.item()))
    

if __name__=="__main__":
    train()
    # validate()