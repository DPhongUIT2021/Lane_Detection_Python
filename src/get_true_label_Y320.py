
import json
import numpy as np


# Specify the file path for the JSON file
name = "label"
#file_path_out = f"json/predict_{name}.json"
file_path_out = f"json/true_{name}_Y320.json"
data_list = []

json_gt = [json.loads(line) for line in open(f'json/true_{name}.json')]
json_gt = json_gt[0]

y_samples = json_gt[0]['h_samples']
len_y_samples = len(y_samples)
start = y_samples.index(320)
y_samples_new = y_samples[start:len_y_samples]

for i in range(0, len(json_gt)):
    gt = json_gt[i] 
    raw_file = gt['raw_file']
    gt_lanes = gt['lanes']

    gt_lanes_left = gt_lanes[0][start:len_y_samples]
    gt_lanes_right = gt_lanes[1][start:len_y_samples]

    gt_lanes_new = np.array([gt_lanes_left, gt_lanes_right])
    gt_lanes_new = gt_lanes_new.tolist()

    data = {
        "lanes": gt_lanes_new,
        "h_samples": y_samples_new,
        "raw_file": raw_file
    }
    data_list.append(data)
            
with open(file_path_out, "w") as json_file:
    json.dump(data_list, json_file)

print("done")




