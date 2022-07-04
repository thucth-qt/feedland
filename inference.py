#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
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

# label_dict = {
#     "empty": 0.0,
#     "minimal": 0.1,
#     "normal": 0.2,
#     "full": 0.3,
# }

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

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    acc = 0
    count = 0
    LABELS = torch.tensor([])
    OUTPUTS = torch.tensor([])
    for idx, loader in enumerate(loaders):
        outputs = []
        with torch.no_grad():
            for batch_idx, (input, _) in enumerate(loader):
                input = input.cuda()
                output = model(input)
                # topk = output.topk(k)[1]
                # topk_ids.append(topk.cpu().numpy())
                outputs.extend((torch.squeeze(output.cpu()).numpy()))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))

        label = label_dict[data_paths[idx].split("/")[-1]]
        labels = torch.tensor([label for i in range(len(outputs))])
        LABELS = torch.cat((LABELS, labels))
        OUTPUTS = torch.cat((OUTPUTS, torch.tensor(outputs)))
        acc += accuracy_reg(torch.unsqueeze(torch.tensor(outputs),1), torch.unsqueeze(labels,1), label_dict).item()*len(outputs)
        count += len(outputs)

        filenames = loader.dataset.filenames()
        with open(result_file, 'a') as out_file:
            for filename, output in zip(filenames, outputs):
                out_file.write('{0},{1}\n'.format(os.path.join(data_paths[idx],filename), output))

    loss_ = F.mse_loss(OUTPUTS, LABELS)
    print("Score: ", acc / count)
    print("MSELoss: ", loss_)


if __name__ == '__main__':
    main()
