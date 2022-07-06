#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
from cv2 import threshold
import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging, accuracy_reg
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

DICT_CLASSNAME = {
    "empty": 0,
    "full": 1,
    "minimal": 2,
    "normal": 3
}


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--label-name', type=str, metavar='class', default='',
                    help='specific class to inference or all class in dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')

label_dict = {
    "empty": 0,
    "minimal": 1,
    "normal": 2,
    "full": 3,
}

def main():
    setup_default_logging()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, './regression_result.csv')
    if os.path.isfile(result_file):
        os.remove(result_file)
    data_paths = []
    if args.label_name != '':
        data_paths.append(os.path.join(args.data, args.label_name))
    else:
        for label in  os.listdir(args.data):
            if label in label_dict.keys(): data_paths.append(os.path.join(args.data, label))
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()
    loaders = []
    for data_path in data_paths:
        loader = create_loader(
            ImageDataset(data_path),
            input_size=config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=True,
            interpolation=config['interpolation'],
            mean=config['mean'],
            std=config['std'],
            num_workers=args.workers,
            crop_pct=1.0 if test_time_pool else config['crop_pct'])
        loaders.append(loader)

    model.eval()

    batch_time = AverageMeter()
    end = time.time()
    OUTPUTS_DICT = {}
    for idx, loader in enumerate(loaders):
        OUTPUTS = torch.tensor([])
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(loader):
                input = input.cuda()
                output = model(input)
                OUTPUTS = torch.cat((OUTPUTS, torch.squeeze(output.cpu())))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        class_ = data_paths[idx].split("/")[-1]
        OUTPUTS_DICT[class_] = OUTPUTS
    
    threshold_range = np.linspace(0, 1, 100, endpoint=False)

    print(OUTPUTS_DICT.keys())

    THRESHOLDS = []
    CLASSES = ['empty', 'minimal', 'normal', 'full']

    i = 0; j = 1
    for idx in range(len(CLASSES) - 1):
        best_acc = 0
        best_thresold = 0
        for threshold in threshold_range:
            threshold = threshold + i
            preds_i = OUTPUTS_DICT[CLASSES[i]]
            preds_j = OUTPUTS_DICT[CLASSES[j]]
            total_i = len(preds_i)
            total_j = len(preds_j)
            correct_i =  (preds_i < threshold).sum().item()
            correct_j =  (preds_j >= threshold).sum().item()
            acc = (correct_i + correct_j) / (total_i + total_j)
            # acc_i = correct_i / total_i
            # acc_j = correct_j / total_j
            # acc = (2 * acc_i * acc_j) / (acc_i + acc_j)

            if acc > best_acc:
                best_acc = acc
                best_thresold = threshold
        i += 1; j+=1
        
        THRESHOLDS.append(best_thresold)

    print(THRESHOLDS)

    return THRESHOLDS


if __name__ == '__main__':
    main()
