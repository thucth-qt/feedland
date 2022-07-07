import sys, os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import os
import glob
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from timm.utils.metrics import accuracy_threshold
from model import DeepRegression
from dataset import data_transform

DATA_DIR = "/content/feedlane/data/classified_data"
OUTPUT_DIR = "/content/feedlane/output"
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def predict(img_path, ckpt_path="/content/feedlane/output/train/lightning_logs/lightning_logs/top_val_loss_models/epoch=92-step=13950.ckpt", thresholds = [0.5,1.5,2.5]):
    # thresholds = [0.48, 1.44, 2.35]
    # thresholds = [0.43, 1.44, 2.07]

    model = DeepRegression.load_from_checkpoint(ckpt_path)
    model.eval()

    img = Image.open(img_path)
    img = data_transform['val'](img)
    # import pdb;pdb.set_trace()
    pred = model(img).detach().numpy().item()
    class_ = None
    if pred < thresholds[0]: class_ = 'empty'
    elif pred < thresholds[1]: class_ = 'minimal'
    elif pred < thresholds[2]: class_ = 'normal'
    else: class_ = 'full'
    
    print("----------------------------------")
    print("Img name: ", img_path)
    print("Pred: ", pred)
    print("Class: ", class_)

if __name__=="__main__":
    img_path = ""
    predict(img_path, "/content/feedlane/output/train/lightning_logs/lightning_logs/top_val_loss_models/epoch=92-step=13950.ckpt")