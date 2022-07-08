from glob import glob
import os 
import shutil

path = "/Users/conglinh/document/FreeLance/feedlane/data/labelstudio"
region_label_filepath = "dataset/uniq_results_label/result.json"
output_dir = "/Users/conglinh/document/FreeLance/feedlane/data/crop_data"

shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)

dataset_root = path 
path_raw_ds = path

uniq_camera = {}
img_paths = glob(os.path.join(path, "*"))

START_AT = "10_162"
END_AT = "_2022"


def get_ip(img_path, start_at, end_at, count_from=0):
    img_path = img_path[img_path.find(start_at)+count_from:]
    ip = img_path[:img_path.find(end_at)]
    return ip
for path in img_paths:
    basename = os.path.basename(path)
    ip = get_ip(basename, START_AT, END_AT)
    uniq_camera[ip] =path  

import os
import glob
import cv2
import json
import numpy as np
from tqdm import tqdm

catid2side = {0:"LEFT", 1:"RIGHT"}

file = open(region_label_filepath)
json_parser = json.load(file)

masks = {}


imgid2ip ={}
for img in json_parser["images"]:
    imgid = img["id"]
    img_path = img["file_name"]

    ip = get_ip(img_path, START_AT, END_AT)
    imgid2ip[imgid] = ip


imgid2bbox = {}
for anno in json_parser["annotations"]:
    if anno["image_id"] not in imgid2bbox:
        imgid2bbox[anno["image_id"]] = {}
    segment = anno["segmentation"][0]
    segment = [int(x) for x in segment]
    segment = list(zip(segment[0::2], segment[1::2]))

    imgid2bbox[anno["image_id"]][catid2side[anno["category_id"]]] = segment

ip2bbox={}
for imgid in imgid2bbox.keys():
    ip2bbox[imgid2ip[imgid]] = imgid2bbox[imgid]

from glob import glob
import cv2 
import matplotlib.pyplot as plt 

for img_filepath in tqdm(glob(os.path.join(path_raw_ds,"*"))[:10], "Cropping images"):
    img = cv2.imread(img_filepath)
    print(img_filepath)
    ip = get_ip(img_filepath, START_AT, END_AT)
    bbox_lr = ip2bbox[ip]
    for side in ["LEFT", "RIGHT"]:
        
        bbox = np.array(bbox_lr[side]).astype(np.int32)
        print(bbox)
        # import pdb; pdb.set_trace()
        cv2.polylines(img, [bbox], isClosed=True, color=(0,255,0), thickness=2)
    # plt.imshow(img[:,:,::-1])
    # plt.show()

def find_center(p1, p2):
    return (p1[0]+p2[0])//2, (p1[1]+p2[1])//2

def rearange(pts):
    top_points = sorted(pts,key=lambda x:x[1])[:2]
    tl_point = sorted(top_points,key=lambda x:x[0])[0]
    tr_point = sorted(top_points,key=lambda x:x[0])[1]
    
    bottom_points = sorted(pts,key=lambda x:x[1])[2:]
    bl_point = sorted(bottom_points,key=lambda x:x[0])[0]
    br_point = sorted(bottom_points,key=lambda x:x[0])[1]
    return [tl_point, tr_point, br_point, bl_point]

def _repeat_to_enough(list_, num, ref_val=None):
    if ref_val is None:
        ref_val = list_[-1]
    rest_len = num-len(list_)
    if rest_len<=0:
        return list_
    return list_ + [list_[-1]]*rest_len

def prevent_zero(x):
    if x==0:
        return 1
    return x
def split_column(bbox, col=2):
    if col<=1:
        return [bbox]
        
    top_mid_points_x = list(range(bbox[0][0], bbox[1][0], prevent_zero((bbox[1][0]-bbox[0][0])//col)))
    top_mid_points_x.append(bbox[1][0])
    top_mid_points_x = _repeat_to_enough(top_mid_points_x, col+1)
    top_mid_points_y = list(range(bbox[0][1], bbox[1][1], prevent_zero((bbox[1][1]-bbox[0][1])//col)))
    top_mid_points_y.append(bbox[1][1])
    top_mid_points_y = _repeat_to_enough(top_mid_points_y, col+1)
    top_mid_points = list(zip(top_mid_points_x, top_mid_points_y))

    bot_mid_points_x = list(range(bbox[3][0], bbox[2][0], prevent_zero((bbox[2][0]-bbox[3][0])//col)))
    bot_mid_points_x.append(bbox[2][0])
    bot_mid_points_x = _repeat_to_enough(bot_mid_points_x, col+1)
    bot_mid_points_y = list(range(bbox[3][1], bbox[2][1], prevent_zero((bbox[2][1]-bbox[3][1])//col)))
    bot_mid_points_y.append(bbox[2][1])
    bot_mid_points_y = _repeat_to_enough(bot_mid_points_y, col+1)
    bot_mid_points = list(zip(bot_mid_points_x, bot_mid_points_y))

    bboxes = []
    for i in range(col):
        bboxes.append((top_mid_points[i], top_mid_points[i+1], bot_mid_points[i+1], bot_mid_points[i]))
    return bboxes


def split_row(bbox, row=2):
    if row<=1:
        return [bbox]
    # import pdb; pdb.set_trace()
    left_mid_points_x = list(range(bbox[0][0], bbox[3][0], prevent_zero((bbox[3][0]-bbox[0][0])//row)))
    left_mid_points_x.append(bbox[3][0])
    left_mid_points_x = _repeat_to_enough(left_mid_points_x, row+1)

    left_mid_points_y = list(range(bbox[0][1], bbox[3][1], prevent_zero((bbox[3][1]-bbox[0][1])//row)))
    left_mid_points_y.append(bbox[3][1])
    left_mid_points_y = _repeat_to_enough(left_mid_points_y, row+1)

    left_mid_points = list(zip(left_mid_points_x, left_mid_points_y))

    right_mid_points_x = list(range(bbox[1][0], bbox[2][0], prevent_zero((bbox[2][0]-bbox[1][0])//row)))
    right_mid_points_x.append(bbox[2][0])
    right_mid_points_x = _repeat_to_enough(right_mid_points_x, row+1)
    right_mid_points_y = list(range(bbox[1][1], bbox[2][1], prevent_zero((bbox[2][1]-bbox[1][1])//row)))
    right_mid_points_y.append(bbox[2][1])
    right_mid_points_y = _repeat_to_enough(right_mid_points_y, row+1)
    right_mid_points = list(zip(right_mid_points_x, right_mid_points_y))

    bboxes = []
    for i in range(row):
        bboxes.append((left_mid_points[i], right_mid_points[i], right_mid_points[i+1], left_mid_points[i+1]))
    return bboxes


def split(bbox, row=5, column=2):
    bbox = rearange(bbox)
    bbox = np.array(bbox).astype(np.int32)
    return_bboxes =[]
    #split columns
    bboxes_col = split_column(bbox, column)

    for bbox in bboxes_col:
        bboxes_row = split_row(bbox, row)
        return_bboxes.extend(bboxes_row)
    return return_bboxes

import os
import cv2
import numpy as np


def rearange(pts):
    # pts = pts.tolist()
    top_points = sorted(pts, key=lambda x: x[1])[:2]
    tl_point = sorted(top_points, key=lambda x: x[0])[0]
    tr_point = sorted(top_points, key=lambda x: x[0])[1]

    bottom_points = sorted(pts, key=lambda x: x[1])[2:]
    bl_point = sorted(bottom_points, key=lambda x: x[0])[0]
    br_point = sorted(bottom_points, key=lambda x: x[0])[1]

    return np.array([tl_point, tr_point, br_point, bl_point], dtype=np.float32)


def crop_and_rotate(img, pts, out_file_path, out_size=(224, 224)):
    if img is None:
        return
    pts = rearange(pts)
    # tl_point, tr_point, br_point, bl_point = pts
    w, h = out_size

    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    warped_img = cv2.warpPerspective(img, M, out_size)
    cv2.imwrite(out_file_path, warped_img)
    # warped = imutils.rotate_bound(warped, 90)
    return


def cut_img(img_path, out_dir_path, row, col, out_size=(64, 112)):
    img = cv2.imread(img_path)
    img_shown = img.copy()
    img_name = os.path.basename(img_path).split(".")[0]

    ip = get_ip(img_path, START_AT, END_AT)
    lst_pts = [ip2bbox[ip]["LEFT"], ip2bbox[ip]["RIGHT"]]
    
    # import pdb; pdb.set_trace()
    lst_pts_split = []
    lst_pts_split.extend(split(lst_pts[0], row=row, column=col))
    lst_pts_split.extend(split(lst_pts[1], row=row, column=col))

    for i, pts in enumerate(lst_pts_split):
        out_file_path = os.path.join(out_dir_path, f"{img_name}_{i}.jpg")
        crop_and_rotate(img, np.float32(pts), out_file_path, out_size)
        #viz
        # import pdb; pdb.set_trace()
        pts = np.array(pts, np.int32)
        cv2.polylines(img_shown, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    # plt.imshow(img_shown[:,:,::-1])
    # plt.show()


if __name__ == "__main__":
    plt.figure(figsize=(15, 40))
    ds_paths = glob(os.path.join(path_raw_ds,"*"))
    for path in ds_paths:
        cut_img(path, \
            output_dir, \
            4, \
            2, \
            out_size=(48, 72))

    # import os, glob, shutil
    # crop_data = "/Users/conglinh/document/FreeLance/feedlane/data/crop_data"
    # output_dir = "/Users/conglinh/document/FreeLance/feedlane/data/classified_data/empty"
    # ls_file_crop = glob.glob(os.path.join(crop_data, "*"))
    # start_withs = []
    # img_paths = []
    # for start_with in start_withs:
    #     img_paths.extend([x for x in ls_file_crop if os.path.basename(x).startswith(start_with)])

    # print(f"Move {len(img_paths)} file to empty")

    # for img_path in img_paths:
    #     output_path = os.path.join(output_dir,os.path.basename(img_path))
    #     # print(img_path + "===>" + output_path)
    #     shutil.move(img_path, output_path)

