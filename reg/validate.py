import sys, os
import glob

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from PIL import Image

from config import FeedlaneConfig
from metric import accuracy_threshold
from model import DeepRegression
from dataset import data_transform

TEST_DIR = os.path.join(FeedlaneConfig.OUTPUT_DIR, "test")
os.makedirs(TEST_DIR, exist_ok=True)

def validate(ckpt_path, thresholds=[0.5,1.5,2.5]):
    assert len(thresholds) == 3
    # thresholds = [0.48, 1.44, 2.35]
    # thresholds = [0.43, 1.44, 2.07]
    result_file = os.path.join(TEST_DIR, 'regression_result.csv')
    if os.path.isfile(result_file):
        os.remove(result_file)

    model = DeepRegression.load_from_checkpoint(ckpt_path)
    # model.set_transform(transforms.ToTensor())
    model.eval()

    TARGETS = []
    PREDS = []
    PREDS_DICT = {}
    for classname in FeedlaneConfig.CLASSNAMES:
        # classname = ""
        SUB_TARGETS = []
        SUB_PREDS = []
        FILES = []
        
        for f in glob.glob(os.path.join(FeedlaneConfig.DATA_VAL_DIR, classname, "*.jpg")):
            SUB_TARGETS.append(FeedlaneConfig.LABEL_DICT[classname])
            # img = torch.tensor(cv2.imread(file))
            img = Image.open(f)
            img = data_transform['val'](img)
            # import pdb;pdb.set_trace()
            pred = model(img).detach().numpy().item()
            SUB_PREDS.append(pred)
            FILES.append(f)
        # import pdb;pdb.set_trace()
        loss_ = F.mse_loss(torch.tensor(SUB_PREDS), torch.tensor(SUB_TARGETS))
        acc_ = accuracy_threshold(torch.tensor(SUB_PREDS), torch.tensor(SUB_TARGETS), FeedlaneConfig.LABEL_DICT, thresholds)
        print("{0} MSELoss: {1}".format(classname, loss_))
        print("{0} ACC: {1}".format(classname, acc_))
        TARGETS.extend(SUB_TARGETS)
        PREDS.extend(SUB_PREDS)
        PREDS_DICT[classname] = SUB_PREDS


        with open(result_file, 'a') as out_file:
            for filename, output in zip(FILES, SUB_PREDS):
                out_file.write('{0},{1}\n'.format(filename, output))
    
    loss_ = F.mse_loss(torch.tensor(PREDS), torch.tensor(TARGETS))
    acc = accuracy_threshold(torch.tensor(PREDS), torch.tensor(TARGETS), FeedlaneConfig.LABEL_DICT, thresholds)
    
    print("----------------------------------")
    print("\n\nTotal ACC: ", acc)
    print("Total MSELoss: ", loss_)

    return PREDS_DICT

if __name__=="__main__":
    validate(ckpt_path=FeedlaneConfig.CKPT_PATH, thresholds=[0.35000000000000003, 1.28, 2.07])