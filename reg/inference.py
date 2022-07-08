import sys, os

import torch
from PIL import Image
from feedlane.reg.config import FeedlaneConfig

from model import DeepRegression
from dataset import data_transform

def predict(img_path, ckpt_path, thresholds=[0.5,1.5,2.5]):
    assert len(thresholds) == 3
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
    predict(img_path, ckpt_path=FeedlaneConfig.CKPT_PATH)