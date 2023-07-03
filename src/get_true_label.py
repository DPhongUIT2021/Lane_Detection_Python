import os
import json
import numpy as np
import copy
from lane_lib_TU import LaneEval
import math

# Specify the file path for the JSON file
name = "0531"
#file_path_out = f"json/predict_{name}.json"
file_path_out = f"json/true_{name}.json"
data_list = []

json_gt = [json.loads(line) for line in open('json/true_{name}.json')]
y_samples = json_gt[0]['h_samples']
size_sample = len(y_samples)

for i in range(0, len(json_gt)):
    gt = json_gt[i] 
    raw_file = gt['raw_file']

    #folder_clips = raw_file[15:19]
    #if folder_clips == name:
    gt_lanes = gt['lanes']
    gt_lanes_new = copy.deepcopy(gt_lanes)
    for i in range(0, len(gt_lanes)):
        arr_tmp = gt_lanes[i]
        count = arr_tmp.count(-2)
        count = size_sample - count
        gt_lanes_new[i].append(count)
    gt_lanes_two_lane_max = sorted(gt_lanes_new, key=lambda x:x[size_sample], reverse=True)

    # get 2 lane with greater point
    gt_lanes_01 = gt_lanes_two_lane_max[0][0:size_sample]
    gt_lanes_02 = gt_lanes_two_lane_max[1][0:size_sample]

    # check get left righ lane
    first_index = gt_lanes_01.index(next(x for x in gt_lanes_01 if x > 0))
    last_index = -1  # Initialize the last index variable
    # Iterate over the array in reverse order
    for i in range(len(gt_lanes_01)-1, -1, -1):
        if gt_lanes_01[i] > 0:
            last_index = i
            break  # Exit the loop once a positive value is found

    if (gt_lanes_01[first_index] - gt_lanes_01[last_index]) > 0:
        gt_lanes_left = gt_lanes_01
        gt_lanes_right = gt_lanes_02
    else:
        gt_lanes_right = gt_lanes_01
        gt_lanes_left = gt_lanes_02

    angles = LaneEval.get_angle(np.array(gt_lanes_left), np.array(y_samples))
    angles_left = math.degrees(angles) + 90

    angles = LaneEval.get_angle(np.array(gt_lanes_right), np.array(y_samples))
    angles_right = math.degrees(angles) + 90

    if (angles_left >= 28 and angles_left <= 66) and (angles_right >= 115 and angles_right <= 151):

        gt_lanes_new = np.array([gt_lanes_left, gt_lanes_right])
        gt_lanes_new = gt_lanes_new.tolist()

        data = {
            "lanes": gt_lanes_new,
            "h_samples": y_samples,
            "raw_file": raw_file
        }
        data_list.append(data)
            
print('len_gt_origin',len(json_gt))
print('len_data_list',len(data_list))

with open(file_path_out, "w") as json_file:
    json.dump(data_list, json_file)

print("done")




