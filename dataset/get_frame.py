from operator import contains
import os, sys
import re
import csv
from telnetlib import IP
from time import sleep
import cv2
import numpy as np
import time

PYTHON_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PYTHON_PATH)
IPS_NOT_PINGABLE = ['10.162.33.50', '10.162.33.51', '10.162.33.52', '10.162.33.53', '10.162.33.54', '10.162.33.55', '10.162.33.56']

def get_frame_manual(num_frame, time_interval, data_folder="./data/images"):
    IPS_ALL = {
        "10.162.33.50",
        "10.162.33.51",
        "10.162.33.52",
        "10.162.33.53",
        "10.162.33.54",
        "10.162.33.55",
        "10.162.33.56",
        "10.162.33.57",
        "10.162.33.58",
        "10.162.33.59",
        "10.162.33.60",
        "10.162.33.61",
        "10.162.33.62",
        "10.162.33.63",
        "10.162.33.64",
        "10.162.33.65",
        "10.162.33.66",
        "10.162.33.67",
        "10.162.33.68",
        "10.162.33.69",
        "10.162.33.70",
        "10.162.33.71",
        "10.162.33.72",
        "10.162.33.73",
        "10.162.33.74",
        "10.162.33.75",
        "10.162.33.76",
        "10.162.33.77",
        "10.162.33.78",
        "10.162.33.79",
        "10.162.33.80",
        "10.162.33.81",
        "10.162.33.82",
        "10.162.33.83",
        "10.162.33.84",
        "10.162.33.85",
        "10.162.33.86",
        "10.162.33.87",
        "10.162.33.88",
        "10.162.33.89",
        "10.162.33.90",
        "10.162.33.91",
        "10.162.33.92",
        "10.162.33.93",
        "10.162.33.94",
        "10.162.33.95",
        "10.162.33.96",
        "10.162.33.97",
        "10.162.33.98",
        "10.162.33.99",
        "10.162.33.100",
        "10.162.33.101",
        "10.162.33.102",
        "10.162.33.103",
        "10.162.33.104",
        "10.162.33.105",
        "10.162.33.106",
        "10.162.33.107",
        "10.162.33.108",
        "10.162.33.109",
        "10.162.33.110",
        "10.162.33.111",
        "10.162.33.112",
        "10.162.33.113"
    }
    IPS_FEEDLANE = {
        "10.162.33.58",
        "10.162.33.62",
        "10.162.33.65",
        "10.162.33.72",
        "10.162.33.79",
        "10.162.33.83",
        "10.162.33.86",
        "10.162.33.90",
        "10.162.33.97",
        "10.162.33.101",
        "10.162.33.104",
        "10.162.33.111",
    }

    IPS_FEEDLANE_NEW = {
        "10.162.33.51",
        "10.162.33.55",
        "10.162.33.68",
        "10.162.33.108",
    }

    IPS_BEDDING = {
        "10.162.33.57",
        "10.162.33.58",
        "10.162.33.59",
        "10.162.33.61",
        "10.162.33.66",
        "10.162.33.69",
        "10.162.33.73",
        "10.162.33.75",
        "10.162.33.87",
        "10.162.33.96",
        "10.162.33.98",
        "10.162.33.103",
        "10.162.33.107",
        "10.162.33.110",
        "10.162.33.112",
    }

    # IPS = IPS_ALL - IPS_FEEDLANE - IPS_BEDDING
    IPS = IPS_FEEDLANE
        

    # for ip in IPS:
    #     res_ping = os.system("ping -c 2 " + ip)
    #     if res_ping == 0:
    #         SELECTED_IPS.append(ip)

    SELECTED_IPS = IPS
    CONN_POOL = {}
    for ip in SELECTED_IPS:
        camera_url = f"rtsp://admin:123456@{ip}:554/unicast/c1/s1/live"
        CONN_POOL[ip] = cv2.VideoCapture(camera_url)
        os.makedirs(os.path.join(data_folder,ip.replace(".", "_")), exist_ok=True)
    
    while num_frame > 0:
        for ip in SELECTED_IPS:
            frame = None
            conn = CONN_POOL[ip]
            try:
                for i in range(10):
                    if conn.isOpened():
                        ret, frame = conn.read()
                    else:
                        camera_url = f"rtsp://admin:123456@{ip}:554/unicast/c1/s1/live"
                        CONN_POOL[ip] = cv2.VideoCapture(camera_url)
            except: pass
            if frame is None:
                continue
            file_path = os.path.join(data_folder,ip.replace('.', '_'),f"{ip.replace('.', '_')}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.jpg")
            cv2.imwrite(file_path, frame)
        sleep(time_interval)
        num_frame -= 1

    for conn in list(CONN_POOL.values()):
        conn.release()

def move_file(src_path, dst_path):
    import shutil
    import os

    files = []
    for r, d, f in os.walk(src_path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))

    for f in files:
        shutil.move(f, dst_path)
if __name__=="__main__":
    get_frame_manual(1,0)
    move_file("/Users/conglinh/document/FreeLance/feedlane/data/images", "/Users/conglinh/document/FreeLance/feedlane/data/labelstudio")
