import argparse
from os import pardir
from pathlib import Path
import glob
import os 
import shutil
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import cv2

ORIGIN_IMG_DIR = "/Users/conglinh/document/FreeLance/feedlane/data/labelstudio"

class History:
    def __init__(self, max_depth=5):
        self.hist=deque()
        self.max_depth=max_depth
    def push(self, data):
        self.hist.append(data)
        if self.hist.__len__()>self.max_depth:
            self.hist.popleft()
    def pop(self):
        return self.hist.pop()

def press(event):
    global key
    key = ord(event.key)
    return key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="output/")
    parser.add_argument("--partterns", nargs="+", required=True, help=" exp: --partterns *.jpg *.png *.jpeg")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    labels =["empty", "minimal", "normal", "full", "noise"]
    [os.makedirs(os.path.join(output_dir, label), exist_ok=True) for label in labels]
    image_paths = []
    for pattern in args.partterns:
        image_paths += list(input_dir.glob(pattern))

    history=History(max_depth=10)
    for image_path in image_paths:
        tailname = "_" + os.path.basename(image_path).split("_")[-1]
        basename = os.path.basename(image_path).replace(tailname, ".jpg")
        origin_img_path = os.path.join(ORIGIN_IMG_DIR, basename)
        # print(image_path)
        # print(origin_img_path)
        image = cv2.imread(str(image_path))
        f = plt.figure(figsize=(8, 8))
        f.add_subplot(2,1,2)
        plt.imshow(cv2.imread(origin_img_path)[:,:,::-1])
        f.add_subplot(2,1,1)
        plt.imshow(cv2.resize(image, dsize=(0, 0), fx=150 / image.shape[1], fy=350 / image.shape[0])[:,:,::-1])
        # plt.show()
        plt.gcf().canvas.mpl_connect('key_press_event', press)        
        while not plt.waitforbuttonpress(): pass
        # print("You pressed: ", key)        
        plt.close('all') 

        # cv2.imshow(str(image_path), cv2.resize(image, dsize=(0, 0), fx=150 / image.shape[1], fy=350 / image.shape[0]))

        # key = cv2.waitKey(0) & 0xFF
        # cv2.destroyAllWindows()
        # empty
        if key  == ord("d"):
            output_path = os.path.join(output_dir,labels[0],os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            print(f"Move {image_path} ======> {output_path}")
            continue
        # minimal
        elif key == ord("f"):
            output_path = os.path.join(output_dir,labels[1],os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            print(f"Move {image_path} ======> {output_path}")
            continue
        # normal
        elif key == ord("j"):
            output_path = os.path.join(output_dir,labels[2],os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            print(f"Move {image_path} ======> {output_path}")
            continue
        # full
        elif key == ord("k"):
            output_path = os.path.join(output_dir, labels[3], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            print(f"Move {image_path} ======> {output_path}")
            continue
        elif key == ord("r"):
            "reverse previous step"
            image_path, output_path = history.pop()
            shutil.move(output_path, image_path)
            continue
        # exit
        elif key == ord("x"):
            break
                # noise
        else:
            output_path = os.path.join(output_dir, labels[4], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            print(f"Move {image_path} ======> {output_path}")
            continue

