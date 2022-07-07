import sys, os
import glob

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from timm.utils.metrics import accuracy_threshold
from model import DeepRegression
from dataset import data_transform

DATA_DIR = "/content/feedlane/data/classified_data"
OUTPUT_DIR = "/content/feedlane/output"
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

def validate(ckpt_path="/content/feedlane/output/train/lightning_logs/lightning_logs/top_val_loss_models/epoch=92-step=13950.ckpt"):
    thresholds = [0.5,1.5,2.5]
    # thresholds = [0.48, 1.44, 2.35]
    # thresholds = [0.43, 1.44, 2.07]
    result_file = os.path.join(OUTPUT_DIR, 'test/regression_result.csv')
    if os.path.isfile(result_file):
        os.remove(result_file)

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
    TARGETS = []
    PREDS = []
    PREDS_DICT = {}
    for class_ in classes:
        # class_ = ""
        SUB_TARGETS = []
        SUB_PREDS = []
        FILES = []
        for f in glob.glob(os.path.join("{0}/val/{1}".format(DATA_DIR,class_), "*.jpg")):
            SUB_TARGETS.append(label_dict[class_])
            # img = torch.tensor(cv2.imread(file))
            img = Image.open(f)
            img = data_transform['val'](img)
            # import pdb;pdb.set_trace()
            pred = model(img).detach().numpy().item()
            SUB_PREDS.append(pred)
            FILES.append(f)
        # import pdb;pdb.set_trace()
        loss_ = F.mse_loss(torch.tensor(SUB_PREDS), torch.tensor(SUB_TARGETS))
        acc_ = accuracy_threshold(torch.tensor(SUB_PREDS), torch.tensor(SUB_TARGETS), label_dict, thresholds)
        print("{0} MSELoss: {1}".format(class_, loss_))
        print("{0} ACC: {1}".format(class_, acc_))
        TARGETS.extend(SUB_TARGETS)
        PREDS.extend(SUB_PREDS)
        PREDS_DICT[class_] = SUB_PREDS


        with open(result_file, 'a') as out_file:
            for filename, output in zip(FILES, SUB_PREDS):
                out_file.write('{0},{1}\n'.format(filename, output))
    
    loss_ = F.mse_loss(torch.tensor(PREDS), torch.tensor(TARGETS))
    acc = accuracy_threshold(torch.tensor(PREDS), torch.tensor(TARGETS), label_dict, thresholds)
    
    print("----------------------------------")
    print("\n\nTotal ACC: ", acc)
    print("Total MSELoss: ", loss_)

    return PREDS_DICT

if __name__=="__main__":
    validate("/content/feedlane/output/train/lightning_logs/lightning_logs/top_val_loss_models/epoch=92-step=13950.ckpt")