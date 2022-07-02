import argparse
from contextlib import ExitStack
from os import pardir
from pathlib import Path
import glob
import os
import shutil
from collections import deque

import cv2


class History:
    def __init__(self, max_depth=5):
        self.hist = deque()
        self.max_depth = max_depth

    def push(self, data):
        self.hist.append(data)
        if self.hist.__len__() > self.max_depth:
            self.hist.popleft()

    def pop(self):
        return self.hist.pop()

    def is_empty(self):
        return self.hist.__len__() == 0

    def len(self):
        return self.hist.__len__()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="output/")
    parser.add_argument("--partterns", nargs="+", required=True, help=" exp: --partterns *.jpg *.png *.jpeg")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    labels = ["empty", "minimal", "normal", "full", "noise"]
    [os.makedirs(os.path.join(output_dir, label), exist_ok=True) for label in labels]
    image_paths = []
    for pattern in args.partterns:
        image_paths += list(input_dir.glob(pattern))
    history = History(max_depth=10)

    i = -1
    while i < len(image_paths)-1:
        i += 1
        image_path = image_paths[i]
        if not os.path.isfile(image_path):
            continue
        image = cv2.imread(str(image_path))
        cv2.imshow(str(image_path), cv2.resize(image, dsize=(0, 0), fx=150 / image.shape[1], fy=400 / image.shape[0]))

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("d"):
            output_path = os.path.join(output_dir, labels[0], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            continue
        elif key == ord("f"):
            output_path = os.path.join(output_dir, labels[1], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            continue
        elif key == ord("j"):
            output_path = os.path.join(output_dir, labels[2], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            continue
        elif key == ord("k"):
            output_path = os.path.join(output_dir, labels[3], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            continue
        elif key == ord("n"):
            output_path = os.path.join(output_dir, labels[4], os.path.basename(image_path))
            shutil.move(image_path, output_path)
            history.push((image_path, output_path))
            continue

        elif key == ord("r"):
            "reverse previous step"
            if history.is_empty():
                continue
            image_path, output_path = history.pop()
            shutil.move(output_path, image_path)
            i -= 2
            continue
        elif key == ord("x"):
            break
