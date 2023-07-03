
import json
import numpy as np
import cv2
#from lane_lib_TU import LaneEval
import video_lib as Vlib
import time
import matplotlib.pyplot as plt

name = "0531_Y320"
file_path_out = f"json/predict_{name}.json"
count_done_max = 100

def main():
    json_gt = [json.loads(line) for line in open(f'json/true_{name}.json')]
    json_gt = json_gt[0]
    y_samples = json_gt[0]['h_samples']
    h_roi = 320;  h = 720; w = 1280; left_x_roi = 658 + 5; right_x_roi = 610 - 10
    w_roi_left = left_x_roi; w_roi_right = w - right_x_roi

    left_point_01 = [416 - 5, h_roi]; left_point_02 = [658 + 5, h_roi]; left_point_03 = [420 + 5, h]; left_point_04 = [0, h]; left_point_05 = [0, 570 -5] 
    right_point_01 = [610 - 5, h_roi]; right_point_02 = [882 + 5, h_roi]; right_point_03 = [w, 540 - 5]; right_point_04 = [w, h]; right_point_05 = [960 - 5, h] 
    
    left_pentagon = np.array([left_point_01, left_point_02, left_point_03, left_point_04, left_point_05]) #[x, y]
    mask_left = np.zeros(shape=(h,w), dtype=np.uint8)
    cv2.fillPoly(mask_left, [left_pentagon], (255))

    right_pentagon = np.array([right_point_01, right_point_02, right_point_03, right_point_04, right_point_05]) #[x, y]
    mask_right = np.zeros(shape=(h,w), dtype=np.uint8)
    cv2.fillPoly(mask_right, [right_pentagon], (255))

    data_list = []
    count_done = 0
    count_img = 0
    for i in range(0, len(json_gt)):
        gt = json_gt[i]
        raw_file = gt['raw_file']
        # if raw_file[-15:-9] != "533319":
        #     continue
        gt_lanes = gt['lanes']
        gt_lanes = np.array(gt_lanes)

        img = cv2.imread(raw_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_left = cv2.bitwise_and(img_gray, mask_left)
        img_right = cv2.bitwise_and(img_gray, mask_right)

        start_time = time.time()

        voting_left = Vlib.sobel_nms_fixT_vote_left(img_left[h_roi:h, 0:left_x_roi])
        voting_right = Vlib.sobel_nms_fixT_vote_right(img_right[h_roi:h, right_x_roi:w]) 

        execution_time = time.time() - start_time
        print(f"Execution time of sobel_nms_fixT_vote_left_right: {execution_time:.6f} seconds")

        predict_lane_left = Vlib.get_arr_x_left(h, w, voting_left, y_samples, gt_lanes[0])
        predict_lane_right = Vlib.get_arr_x_right(h, w, voting_right, y_samples, gt_lanes[1])

        pred_lanes = np.array([predict_lane_left, predict_lane_right])
        pred_lanes = pred_lanes.tolist()

        data = {
            "lanes": pred_lanes,
            "h_samples": y_samples,
            "raw_file": raw_file
        }
        data_list.append(data)

        # if count_done == count_done_max:
        #     break
        # count_done +=1
        # print("done", count_done)

        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        draw_predi_left = [[(x, y) for (x, y) in zip(pred_lanes[0], y_samples) if x >= 0] ]
        draw_predi_right = [[(x, y) for (x, y) in zip(pred_lanes[1], y_samples) if x >= 0] ]

        for point in draw_predi_left:
            #cv2.polylines(img, np.int32([lane]), isClosed=False, color=(255,0,0), thickness=2)
            cv2.circle(img, point, radius=2, color=(0, 0, 255), thickness=-1)
        for point in draw_predi_right:
            #cv2.polylines(img, np.int32([lane]), isClosed=False, color=(255,0,0), thickness=2)
            cv2.circle(img, point, radius=2, color=(0, 0, 255), thickness=-1)

        draw_gt_lane = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
        for lane in draw_gt_lane:
            for pt in lane:
                cv2.circle(img, pt, radius=25, color=(0, 255, 0), thickness=1)
        # for height in y_samples:
        #     start_point = (0, height)  # Starting from the left, at the specified height
        #     end_point = (img.shape[1], height)  # Ending at the right, at the specified height
        #     cv2.line(img, start_point, end_point, (0, 255, 0), 1)

        plt.imshow(img)
        plt.title(str(count_img) + '_' + raw_file[-15:-9])
        plt.show()

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # path_save ='Analysis/img_clips_0531/' + str(count_img) + '_' + raw_file[-15:-9]  + '.jpg'
        # cv2.imwrite(path_save, img)
        print(str(count_img) + '_' + raw_file[-15:-9])
        count_img += 1
        

        
    # with open(file_path_out, "w") as json_file:
    #     json.dump(data_list, json_file)


if __name__ == "__main__":

    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"Execution time DONE: {execution_time:.6f} seconds")

    print("===============done===============")
