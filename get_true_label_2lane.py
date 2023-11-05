import numpy as np
import math
import openpyxl
import cv2
import json
import time
import matplotlib.pyplot as plt
import pickle
import random
import os
import copy
import os
from lib_TU_me import LaneEval
import time
import openpyxl
import json
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_thickness = 1
font_color = (0, 0, 255)  # BGR format (blue, green, red)

wb = openpyxl.Workbook()
ws = wb.active

y_start = 320; y_stop = 450; x_ext_bot = 500; x_ext_top = 200; x_mid = 820-60

def get_true_label_2_multi_lane(name):
    data_list = []
    json_gt = [json.loads(line) for line in open(f'json/true/{name}_origin.json')][0]

    for i in range(0, len(json_gt)):
        gt = json_gt[i] 
        raw_file_new = gt['raw_file_new']
        raw_file_old = gt['raw_file_old']
        lanes = gt['lanes']

        lane_left, lane_right = [], []
        for l in lanes:
            l_y_stop = [[x,y] for x,y in l if y <= y_stop and y >= y_start]
            if len(l_y_stop) == 0: continue
            angles = LaneEval.get_angle(l_y_stop)
            angles = math.degrees(angles)
            
            x_top, y_top = l_y_stop[0]
            x_bot, y_bot = l_y_stop[-1]
            if angles < 0:
                if x_bot < (x_mid - 30):
                    lane_left.append(l)
            elif angles > 0:
                if x_bot > (x_mid + 30):
                    lane_right.append(l)
        multi_lane_left = lane_left
        multi_lane_right = lane_right

        mid_lane_left, mid_lane_right = [], []

        if len(lane_left) >= 2:
            mid_lane_left = lane_left[0]
            for k in range(1, len(lane_left)):
                l1 = mid_lane_left
                l2 = lane_left[k]
                l1 = [[x,y] for x,y in l1 if y <= y_stop and y >= y_start]
                l2 = [[x,y] for x,y in l2 if y <= y_stop and y >= y_start]
                x1_bot, y1_top = l1[-1]; x2_bot, y2_top = l2[-1]
                if x2_bot > x1_bot:
                    mid_lane_left = l2
        elif len(lane_left) == 1:
            mid_lane_left = lane_left[0]

        if len(lane_right) >= 2:
            mid_lane_right = lane_right[0]
            for k in range(1, len(lane_right)):
                l1 = mid_lane_right
                l2 = lane_right[k]
                l1 = [[x,y] for x,y in l1 if y <= y_stop and y >= y_start]
                l2 = [[x,y] for x,y in l2 if y <= y_stop and y >= y_start]
                x1_bot, y1_top = l1[-1]; x2_bot, y2_top = l2[-1]
                if x1_bot < x2_bot:
                    mid_lane_right = l1
        elif len(lane_right) == 1:
            mid_lane_right = lane_right[0]

        angle_mid_lane_left = abs(math.degrees(LaneEval.get_angle(mid_lane_left)))
        angle_mid_lane_right = math.degrees(LaneEval.get_angle(mid_lane_right))
        
        data = {
            "raw_file_new": raw_file_new,
            "raw_file_old" : raw_file_old,
            "lanes": lanes,
            "mid_lane_left": mid_lane_left,
            "mid_lane_right": mid_lane_right,
            "multi_lane_left": multi_lane_left,
            "multi_lane_right": multi_lane_right,
            "angle_mid_lane_left": angle_mid_lane_left,
            "angle_mid_lane_right": angle_mid_lane_right,
        }
        data_list.append(data)
                
    print('len_gt_origin',len(json_gt))
    print('len_data_list',len(data_list))

    with open(f"json/true/{name}.json", "w") as json_file:
        json.dump(data_list, json_file)
    

def create_json_from_txt():
    #file_list_test = 'test0_normal.txt'
    for file_list_test in os.listdir(f'data/list/test_split'):
        file_list_test = file_list_test[:-4]
        #if file_list_test != 'test0_normal': continue
        data_list = []; frame_counter = 0
        if not os.path.exists(f'dataset_phan_loai/{file_list_test}'):
            os.makedirs(f'dataset_phan_loai/{file_list_test}')
        with open(f"data/list/test_split/{file_list_test}.txt", "r") as file:
            lines = file.readlines()
        for line in lines:
            raw_file_old = line.strip()
            raw_file_old = f'dataset_goc/{raw_file_old}'
            raw_file_new = f'dataset_phan_loai/{file_list_test}/{frame_counter:04d}.jpg'
            
            file_txt = f"{raw_file_old[:-4]}.lines.txt"
            if not os.path.exists(file_txt): 
                print(f'{file_txt} -   not exits')

            with open(file_txt, 'r') as file:
                content = file.readlines()
            lanes = []
            for lane in content:
                lane = lane.strip(); lane = lane.split(); k = 0; arr_xy = []
                while k < len(lane):
                    x = round(float(lane[k])); y = lane[k+1]; k = k + 2
                    arr_xy.append((int(x), int(y)))
                lanes.append(arr_xy)

            for i in range(len(lanes)):
                l = lanes[i]
                l = [(x,y) for x, y in l if x > 0]
                l = sorted(l, key=lambda x: x[1], reverse=False)
                lanes[i] = l

            #if frame_counter > 9371:
            # img = cv2.imread(f'{raw_file_old}')
            # cv2.imwrite(f'{raw_file_new}', img)

            data = {
                "raw_file_old": raw_file_old,
                "raw_file_new": raw_file_new,
                'lanes': lanes,
            }
            data_list.append(data)
            frame_counter += 1; print(f'{file_list_test}  -  {raw_file_old} - {raw_file_new}')

        with open(f"json/true/{file_list_test}_origin.json", "w") as json_file:
            json.dump(data_list, json_file)

if __name__ == "__main__":
    #create_json_from_txt()
    for name in os.listdir(f'data/list/test_split'):
        name = name[:-4]
        #name = 'test2_hlight'
        get_true_label_2_multi_lane(name)
    #create_json_from_txt()


    print("===============done===============")
