import os
import cv2
import numpy as np

def rearange(pts):
    pts = pts.tolist()
    top_points = sorted(pts,key=lambda x:x[1])[:2]
    tl_point = sorted(top_points,key=lambda x:x[0])[0]
    tr_point = sorted(top_points,key=lambda x:x[0])[1]
    
    bottom_points = sorted(pts,key=lambda x:x[1])[2:]
    bl_point = sorted(bottom_points,key=lambda x:x[0])[0]
    br_point = sorted(bottom_points,key=lambda x:x[0])[1]

    return np.array([tl_point, tr_point, br_point, bl_point], dtype=np.float32)

def crop_and_rotate(img, pts, out_file_path, out_size=(224,224)):
    if img is None: return
    pts = rearange(pts)
    # tl_point, tr_point, br_point, bl_point = pts
    w, h = out_size

    pts2 = np.float32([[0, 0],[w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    warped_img = cv2.warpPerspective(img, M, out_size)
    cv2.imwrite(out_file_path, warped_img)
    # warped = imutils.rotate_bound(warped, 90)
    return

def cut_img(img_path, lst_pts, out_dir_path, out_size=(224,224)):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path).split(".")[0]
    for i, pts in enumerate(lst_pts):
        out_file_path = os.path.join(out_dir_path,f"{img_name}_{i}.jpg")
        crop_and_rotate(img, np.float32(pts), out_file_path, out_size)

if __name__ == "__main__":
    lst_pts = [
        [
          [830.1097242117702,
          298.99817586503],
          [865.1811992066666,
          298.9981758650301],
          [818.2473466966103,
          407.5040270594216],
          [762.6665178926818,
          407.5040270594216]
        ],
        [
          [1265.6716417910447,
          900.7964601769912],
          [1382.6865671641795,
          903.1858407079644],
          [1456.716417910448,
          1065.6637168141592],
          [1330.1492537313434,
          1065.6637168141592]
        ]
    ]
    cut_img("/content/feedlane/data/images/01bac556-10_162_33_97_20220619_180820.jpg", lst_pts, "/content/feedlane/data")
