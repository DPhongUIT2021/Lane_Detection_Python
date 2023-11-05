# I1: gốc toạ độ Left: góc trên cùng bên phải
#                 Right: góc trên cùng bên trái
# Ưu điểm: giá trị Rho nhỏ, có thể tăng độ chính xác
# Nhược điểm: vì giá Rho nhỏ nên Rho các làn khá gần nhau nên rất khó cho việc tìm Rho bằng cách cộng xung quanh
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
from lib_TU_me import LaneEval

# Define the font properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
font_color = (0, 0, 0)  # BGR format (blue, green, red)

wb = openpyxl.Workbook()
ws = wb.active
#ban dau y_stop = 430
y_start = 320-50; y_stop = 450; x_ext_bot = 500; x_ext_top = 200; x_mid = 820-60
h = 590; w = 1640
h_roi = y_stop - y_start; w_roi = x_ext_bot
y_roi = y_stop - y_start; x_roi = x_ext_bot

phi_min = 0; phi_max = 75

Rho_max_ram = round(math.sqrt((y_stop-y_start)**2 + (x_ext_bot)**2)) # idea 3: origin of coordinate in top leftmost corner, in top rightmost corner
phi_max_ram = phi_max

roi_left = np.array([((w_roi-x_ext_top),0), (w_roi,0), (w_roi,h_roi), (0,h_roi)])
mask_left = np.zeros(shape=(h_roi,w_roi), dtype=np.uint8)
cv2.fillPoly(mask_left, [roi_left], (255))

roi_right = np.array([(0,0), (x_ext_top,0), (w_roi,h_roi), (0,h_roi)])
mask_right = np.zeros(shape=(h_roi,w_roi), dtype=np.uint8)
cv2.fillPoly(mask_right, [roi_right], (255))

roi_left_draw = np.array([((x_mid-x_ext_top),y_start), (x_mid,y_start), (x_mid,y_stop), (x_mid-x_ext_bot,y_stop)])
roi_right_draw = np.array([(x_mid,y_start), (x_mid+x_ext_top,y_start), (x_mid+x_ext_bot,y_stop), (x_mid,y_stop)])


global path_voting
global Flow
global phi_rotation, n_frame, X_rotate, t_g, denta_gray_max, thresh_denta_x, pixel_thresh_width_2dege

###########################################################333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

class Voting:
    def get_xy_phi_from_gray(self, img, flag_left):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3); sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        img_G = np.round(np.sqrt(sobel_x**2 + sobel_y**2)); img_phi = np.round(np.degrees(np.arctan2(sobel_y, sobel_x)))

        img_phi_tmp = np.zeros(shape=(img_phi.shape[0],img_phi.shape[1]))
        for i in range(5, img_phi.shape[0]-5):
            for j in range(5, img_phi.shape[1]-5):
                if img_G[i,j] > t_g:
                    phi = img_phi[i,j]; phi = Voting().check_phi(phi,flag_left)   
                    if phi > 0:
                        img_phi_tmp[i,j] = int(phi)
                    else:
                        img_G[i,j] = 0
        img_phi = img_phi_tmp

        arr_xy_phi = []
        if flag_left:
            i = 5
            while i < img.shape[0]-5:
                j = img.shape[1]-5
                flag_rising = 0; arr_xy_candidate = []; gray_max = 0; note_left = [0, 0, 0]
                while j >= 5:
                    G4 = img_G[i,j-1]; G5 = img_G[i, j]; G6 = img_G[i,j+1]
                    if G5 > G4 and G5 > G6 and G4 > t_g and G6 > t_g:
                        if (img_G[i-1,j+1] > t_g or img_G[i-1,j+2] > t_g) and (img_G[i+1,j-1] > t_g or img_G[i+1,j-2] > t_g):
                            y = i; x = j
                            if Voting().check_rising(img[y,x-2:x+3], flag_left):
                                flag_rising = 1
                                x0 = x
                            elif Voting().check_falling(img[y,x-2:x+3], flag_left) and flag_rising == 1:
                                flag_rising = 0
                                deta_x = abs(x0 - x)
                                p0 = np.mean(img_phi[y,x0-1:x0+2]); p1 = np.mean(img_phi[y,x-1:x+2]); phi = (p0 + p1) / 2.
                                thresh_denta_x = pixel_thresh_width_2dege / math.cos(math.radians(phi))
                                # min_data_x = 3 / math.cos(math.radians(phi))
                                # if deta_x <= thresh_denta_x and deta_x >= min_data_x:
                                if deta_x <= thresh_denta_x:
                                    x1 = int((x + x0) / 2)
                                    mean_gray = np.mean(img[y,x1-1:x1+2])
                                    if gray_max < mean_gray: gray_max = mean_gray
                                    arr_xy_candidate.append((x1, phi, mean_gray))
                    j = j - 1
                for pt in arr_xy_candidate:
                    if gray_max - pt[-1] <= denta_gray_max:
                        if note_left[0] < pt[0]:
                            note_left = pt
                if gray_max != 0:
                    x, phi, mean_gray = note_left
                    arr_xy_phi.append((x, y, phi))
                i = i + 1
        else:
            i = 5
            while i < img.shape[0]-5:
                j = 5
                flag_rising = 0
                arr_xy_candidate = []; gray_max = 0; note_right = [w_roi, 0, 0]
                while j < img.shape[1]-5:
                    G4 = img_G[i,j-1]; G5 = img_G[i, j]; G6 = img_G[i,j+1]
                    if G5 > G4 and G5 > G6 and G4 > t_g and G6 > t_g:
                        if img_G[i-1,j-1] > t_g or img_G[i-1,j-2] > t_g or img_G[i+1,j+1] > t_g or img_G[i+1,j+2] > t_g:
                            y = i; x = j
                            if Voting().check_rising(img[y,x-2:x+3], flag_left):
                                flag_rising = 1
                                x0 = x
                            elif Voting().check_falling(img[y,x-2:x+3], flag_left) and flag_rising == 1:
                                flag_rising = 0
                                deta_x = abs(x0 - x)
                                p0 = np.mean(img_phi[y,x0-1:x0+2]); p1 = np.mean(img_phi[y,x-1:x+2]); phi = (p0 + p1) / 2.
                                thresh_denta_x = pixel_thresh_width_2dege / math.cos(math.radians(phi))
                                # min_data_x = 3 / math.cos(math.radians(phi))
                                # if deta_x <= thresh_denta_x and deta_x >= min_data_x:
                                if deta_x <= thresh_denta_x:
                                    x1 = int((x + x0) / 2)
                                    mean_gray = np.mean(img[y,x1-1:x1+2])
                                    if gray_max < mean_gray: gray_max = mean_gray
                                    arr_xy_candidate.append((x, phi, mean_gray))
                    j = j + 1
                for pt in arr_xy_candidate:
                    if gray_max - pt[-1] <= denta_gray_max:
                        if note_right[0] > pt[0]:
                            note_right = pt
                if gray_max != 0:
                    x, phi, mean_gray = note_right
                    arr_xy_phi.append((x, y, phi))
                i = i + 1
        return arr_xy_phi
    def check_edge_max(self, G5, arrT, arrB, flag_left):
        t1, t2, t3, t4, t5 = arrT[0], arrT[1], arrT[2], arrT[3], arrT[4]
        b1, b2, b3, b4, b5 = arrB[0], arrB[1], arrB[2], arrB[3], arrB[4]
        if flag_left:
            maximum = max(t2,b4)
            minimum = min(t3,t4,b2,b3)
        else:
            maximum = max(t4,b2)
            minimum = min(t2,t3,b3,b4)

        if G5 > maximum and minimum > t_g:
            return True
        else:
            return False
    def check_weak_edge(self, arr_gray):
        mean = np.mean(arr_gray)
        if mean > 127:
            return True       
        else:
            return False
    def check_nms(self,G1, G3, G4, G5, G6, G7, G9, flag_left):
        if flag_left:
            max_G = max(G1*0.875, G9*0.875, G4, G6)
        else:
            max_G = max(G3*0.875, G7*0.875, G4, G6)

        if G5  > max_G:
            return True
        return False
    def check_phi(self,phi,flag_left):
        if flag_left:
            if (phi >= -(180-phi_min) and phi <= -(180-phi_max)) or (phi >= phi_min and phi <= phi_max):
                if phi < 0: phi = 180 + phi
            else:
                return 0
        else:
            if (phi >= -phi_max and phi <= -phi_min) or (phi >= (180-phi_max) and phi <= (180-phi_min)):
                if phi <= 0: phi = abs(phi)
                else: phi = 180 - phi
            else:
                return 0
        return phi
    def check_rising(self, arr, flag_left):
        arr = arr.astype(np.int16)
        if flag_left:
            k = 0
            while(k < len(arr)-1):
                denta = arr[k] - arr[k+1]
                if denta >= 0 or (denta < 0 and denta >= -2):
                    k = k + 1
                else:
                    return False
            return True
        else:
            k = 0
            while(k < len(arr)-1):
                denta = arr[k+1] - arr[k]
                if denta >= 0 or (denta < 0 and denta >= -2):
                    k = k + 1
                else:
                    return False
            return True
    def check_falling(self, arr, flag_left):
        arr = arr.astype(np.int16)
        if flag_left:
            k = 0
            while(k < len(arr)-1):
                denta = arr[k+1] - arr[k]
                if denta >= 0 or (denta < 0 and denta >= -2):
                    k = k + 1
                else:
                    return False
            return True
        else:
            k = 0
            while(k < len(arr)-1):
                denta = arr[k] - arr[k+1]
                if denta >= 0 or (denta < 0 and denta >= -2):
                    k = k + 1
                else:
                    return False
            return True
    def is_white(self, arr_gray):
        mid_len = int(len(arr_gray)/2)
        arr_left = arr_gray[0:mid_len-2]; arr_right = arr_gray[mid_len+3:]
        mean_left = np.mean(arr_left); mean_right = np.mean(arr_right)
        if mean_left >= 200 or mean_right >= 200: return True
        else: return False
    def check_phi_doRho(self, phi, i, j, flag_left):
        if flag_left:
            dy = i; dx = j
        else:
            dy = i; dx = (w_roi-1)-j
        Rho = round(dx*math.cos(math.radians(phi)) + dy*math.sin(math.radians(phi)))  
        if Rho >= 0 and Rho <= Rho_max_ram:
            return Rho, phi
        else: print(f'Rho_lef_or_right: {Rho} in dx,dy,phi: {dx,dy,phi} out of range')
        return -1, -1
    def get_arr_coordiante(self, Rho_phi, flag_left, mid_lane_true):
        arr_x = []
        if len(Rho_phi) == 0:
            Rho_phi = [(0, 5)]
            print(f'No have line')
        for rp in Rho_phi:
            Rho, phi = rp[0], rp[1]
            
            sin_phi = math.sin(math.radians(phi)); cos_phi = math.cos(math.radians(phi))
            Rho = [(Rho*8+i) for i in range(0, 8)] 
            for pt in mid_lane_true:
                x_true, y_true = pt
                dy = y_true - y_start
                dx = [(r - dy * sin_phi) / cos_phi for r in Rho]
                dx = np.mean(dx)
                #dx = (Rho - dy * sin_phi) / cos_phi
                if flag_left:
                    x = round(dx + (x_mid - w_roi))
                    if x >= 0:
                        arr_x.append([int(x), y_true])
                    else: arr_x.append([int(0), y_true])
                else: 
                    x = round((w_roi-dx) + x_mid)
                    if x >= 0 and x <= w:
                        arr_x.append([int(x), y_true])
                    else: arr_x.append([int(w), y_true])
        return arr_x
    def voting_factor_Rho(self,voting_left, voting_right, factor_Rho):
        voting_left_new = []; voting_right_new = []
        for i in range(0, voting_left.shape[0]-factor_Rho, factor_Rho):
            arr_sum_left = np.zeros(voting_left.shape[1], dtype=np.uint16); arr_sum_right = np.zeros(voting_right.shape[1], dtype=np.uint16)
            for i_Rho in range(0, factor_Rho):
                i_tmp = i + i_Rho
                arr_tmp_left = voting_left[i_tmp];  arr_tmp_right = voting_right[i_tmp]
                arr_sum_left += arr_tmp_left; arr_sum_right += arr_tmp_right 
            voting_left_new.append(arr_sum_left); voting_right_new.append(arr_sum_right)
        voting_right_new = np.array(voting_right_new); voting_left_new = np.array(voting_left_new)
        return voting_left_new, voting_right_new
    def voting_plus_around(self,voting_left, voting_right, ext_phi, ext_r):
        voting_left_new = np.zeros_like(voting_left); voting_right_new = np.zeros_like(voting_right)

        for i in range(ext_r, voting_left.shape[0]-ext_r):
            for j in range(ext_phi, voting_left.shape[1]-ext_phi):
                arr_left = voting_left[i-ext_r:i+(ext_r+1), j-ext_phi:j+(ext_phi+1)]
                arr_right = voting_right[i-ext_r:i+(ext_r+1), j-ext_phi:j+(ext_phi+1)]
                voting_left_new[i,j] = np.sum(arr_left) 
                voting_right_new[i,j] = np.sum(arr_right)

        return voting_left_new, voting_right_new
    def update_save_xy_phi(self, name, number_frame_update):
        json_gt = [json.loads(line) for line in open(f"json/Save_XY_Phi/{name}_{Flow}_F{n_frame}.json")][0]; 
        total_frame = len(json_gt); data_list = []

        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_origin = gt['raw_file_origin']

            if int(raw_file_new[-8:-4]) != number_frame_update: continue

            raw_file_origin = f"{raw_file_origin[:-6]}{str(n_frame)}.jpg"
            img = cv2.imread(raw_file_origin); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            img_left = img[y_start:y_stop, 0:x_mid]
            img_right = img[y_start:y_stop, x_mid:w]

            img_left = cv2.medianBlur(img_left, ksize=5)
            img_right = cv2.medianBlur(img_right, ksize=5)

            voting_left = self.get_xy_phi_from_gray(img_left, flag_left=1)
            voting_right = self.get_xy_phi_from_gray(img_right, flag_left=0)
            print(f"{raw_file_new} / {total_frame}:", time.time() - start_time, "seconds") 

            total_point = (len(voting_left), len(voting_right))

            gt['lane_left'] = voting_left; gt['lane_right'] = voting_right; gt['total_point'] = total_point
            with open(f"json/Save_XY_Phi/{name}_{Flow}_F{n_frame}.json", 'w') as file:
                json.dump(json_gt, file)
    def Save_voting_PKL_after_HT(seft, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        total_frame = len(json_gt); data_list = []
   
        total_time = []
        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']
            mid_lane_left = gt['mid_lane_left']; mid_lane_right = gt['mid_lane_right']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue

            img = cv2.imread(raw_file_old); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            img_left = img[y_start:y_stop, (x_mid-x_ext_bot):x_mid]
            img_right = img[y_start:y_stop, x_mid:(x_mid+x_ext_bot)]

            start_time = time.time()
            #img_left = cv2.medianBlur(img_left, ksize=3)
            #img_right = cv2.medianBlur(img_right, ksize=3)
            img_left = cv2.GaussianBlur(img_left, (3, 3), 1.0)
            img_right = cv2.GaussianBlur(img_right, (3, 3), 1.0)

            arr_xy_phi_left = Voting().get_xy_phi_from_gray(img_left, flag_left=1)
            arr_xy_phi_right = Voting().get_xy_phi_from_gray(img_right, flag_left=0)

            total_point = (len(arr_xy_phi_left), len(arr_xy_phi_right))

            voting_left = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_left:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=1)
                            if Rho_vote != -1:
                                voting_left[int(Rho_vote), int(phi_vote)] += 1

            voting_right = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_right:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=0)
                            if Rho_vote != -1:
                                voting_right[int(Rho_vote), int(phi_vote)] += 1

            total_time.append(time.time() - start_time)

            voting_left, voting_right = Voting().voting_factor_Rho(voting_left, voting_right, factor_Rho=8)  
            #voting_left, voting_right = Voting().voting_plus_around(voting_left, voting_right, ext_phi = 2, ext_r = 2)

            #voting_left[0:18,] = 0 ; voting_right[0:18,] = 0 # voi Roi h[290->490] w[x_mid-400,x_mid] x_top = 400/2 

            with open(f"pkl/voting/{name}/left/{raw_file_new[-8:-4]}_p{phi_rotation}_X{X_rotate}.pkl", "wb") as file:
                pickle.dump(voting_left, file)
            with open(f"pkl/voting/{name}/right/{raw_file_new[-8:-4]}_p{phi_rotation}_X{X_rotate}.pkl", "wb") as file:
                pickle.dump(voting_right, file)

            # save_csv(path=f'csv/debug/voting/{name}_{raw_file_new[-8:-4]}_left.csv', img=voting_left)
            # save_csv(path=f'csv/debug/voting/{name}_{raw_file_new[-8:-4]}_right.csv', img=voting_right)
            print(f'{raw_file_new} - s/frame: {round(time.time() - start_time,3)}')  
            
        #     data = {
        #         "raw_file_old": raw_file_old,
        #         "raw_file_new": raw_file_new,
        #         "arr_xy_phi_left": arr_xy_phi_left,
        #         "arr_xy_phi_right": arr_xy_phi_right,
        #         "total_point": total_point,
        #     }
        #     data_list.append(data)

        # with open(f"json/Save_XY_Phi/{name}_Frame[{number_frame[0]},{number_frame[1]}].json", "w") as json_file:
        #     json.dump(data_list, json_file)

        print(f't_g:{t_g}; pixel_thresh_width_2dege:{pixel_thresh_width_2dege}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}')
        print(f'{name}_Frame[{number_frame[0]},{number_frame[1]}] - ms/frame: {round(np.mean(total_time) * 1000, 2)}') 
    def read_voting_PKL_to_end(seft, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        total_frame = len(json_gt); data_list_acc = []
   
        arr_acc_left = []; arr_acc_right = []
        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']
            mid_lane_left = gt['mid_lane_left']; mid_lane_right = gt['mid_lane_right']
            multi_lane_left = gt['multi_lane_left']; multi_lane_right = gt['multi_lane_right']; lanes = gt['lanes']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue

            with open(f"pkl/voting/{name}/left/{raw_file_new[-8:-4]}_p{phi_rotation}_X{X_rotate}_them_median.pkl", "rb") as file:
                voting_left = pickle.load(file)
            with open(f"pkl/voting/{name}/right/{raw_file_new[-8:-4]}_p{phi_rotation}_X{X_rotate}_them_median.pkl", "rb") as file:
                voting_right = pickle.load(file)  

            # rho_xoa = 18
            # voting_left[0:rho_xoa+1,] = 0 ; voting_right[0:rho_xoa+1,] = 0 # voi Roi h[290->450] w[x_mid-400,x_mid] x_top = 400/2 

            Rho_phi_left = Peak_Rho_phi().peak_Rho_phi_01(voting_left)
            Rho_phi_right = Peak_Rho_phi().peak_Rho_phi_01(voting_right)

            pred_lane_left = Voting().get_arr_coordiante(Rho_phi_left, flag_left=1, mid_lane_true=mid_lane_left)
            pred_lane_right = Voting().get_arr_coordiante(Rho_phi_right, flag_left=0, mid_lane_true=mid_lane_right)

            mid_lane_left = [[x,y] for x,y in mid_lane_left if y <= y_stop] 
            mid_lane_right = [[x,y] for x,y in mid_lane_right if y <= y_stop] 
            pred_lane_left = [[x,y] for x,y in pred_lane_left if y <= y_stop] 
            pred_lane_right = [[x,y] for x,y in pred_lane_right if y <= y_stop] 

            angles_left = LaneEval.get_angle(mid_lane_left)
            angles_right = LaneEval.get_angle(mid_lane_right)

            left_thresh = LaneEval.pixel_thresh / np.cos(angles_left)
            right_thresh = LaneEval.pixel_thresh / np.cos(angles_right)

            n_diem_dung_L, acc_L = LaneEval.sum_point_correct(pred_lane_left, mid_lane_left, left_thresh)

            n_diem_dung_R, acc_R = LaneEval.sum_point_correct(pred_lane_right, mid_lane_right, right_thresh)

            # img = cv2.imread(raw_file_old)

            # cv2.polylines(img, [roi_left_draw], isClosed=True, color=(127,255,0), thickness=2)
            # cv2.polylines(img, [roi_right_draw], isClosed=True, color=(255,127,0), thickness=2)
            
            # for lane in lanes:
            #     for pt in lane:
            #         cv2.circle(img, pt, radius=2, color=(255, 0, 255), thickness=-1)

            # for pt in mid_lane_left:
            #     x,y = pt[0], pt[1]
            #     cv2.circle(img, (x,y), radius=2, color=(0, 255, ), thickness=-1)
            #     cv2.line(img, (round(x-left_thresh), y), (round(x+left_thresh), y), (127,127,0), 1)
            # for pt in mid_lane_right:
            #     x,y = pt[0], pt[1]
            #     cv2.circle(img, (x,y), radius=2, color=(0, 255, ), thickness=-1)
            #     cv2.line(img, (round(x-right_thresh), y), (round(x+right_thresh), y), (127,127,0), 1)

            # for pt in pred_lane_left:
            #     cv2.circle(img, pt, radius=4, color=(0, 0, 255), thickness=1)
            # for pt in pred_lane_right:
            #     cv2.circle(img, pt, radius=4, color=(0, 0, 255), thickness=1)

            # x, y, w, h = (460-20)-10, (y_start-50)-10, 700, (50-10)
            # mask = np.zeros_like(img)
            # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # img = np.where(mask == 255, (255, 255, 255), img)

            # # str_tmp = f't_g:{t_g}; pixel_thresh_width_2dege:{pixel_thresh_width_2dege}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}'
            # # cv2.putText(img, str_tmp, (460,y_start-50), font, font_scale, font_color, font_thickness)
            # str_tmp = f'left: {n_diem_dung_L}/{len(mid_lane_left)} - {round(acc_L, 2)}%             right: {n_diem_dung_R}/{len(mid_lane_right)} - {round(acc_R, 2)}%'
            # cv2.putText(img, str_tmp, (460,y_start-25), font, font_scale, font_color, font_thickness)


            # img = img[y_start-70:y_stop+20, x_mid-x_ext_bot-20:x_mid+x_ext_bot+20, ]
            # cv2.imwrite(f'dataset_phan_loai/{name}/{raw_file_new[-8:]}', img)

            arr_acc_left.append(acc_L)
            arr_acc_right.append(acc_R)
            print(f'{raw_file_new} - left: {n_diem_dung_L}/{len(mid_lane_left)} - {round(acc_L, 2)}, right: {n_diem_dung_R}/{len(mid_lane_right)} - {round(acc_R, 2)} s/frame: {round(time.time() - start_time,3)}')  
            
            data_acc = {
                "raw_file_old": raw_file_old,
                "raw_file_new": raw_file_new,
                "acc_left": acc_L,
                "ratio_left": f'{n_diem_dung_L}/{len(mid_lane_left)}',
                "acc_right": acc_R,
                "ratio_right": f'{n_diem_dung_R}/{len(mid_lane_right)}',
                "Rho_phi_left": Rho_phi_left,
                "Rho_phi_right": Rho_phi_right,
                "pred_lane_left": pred_lane_left,
                "pred_lane_right": pred_lane_right,
            }
            data_list_acc.append(data_acc)

        with open(f"json/Acc/{name}_Frame[{number_frame[0]},{number_frame[1]}]_them_median.json", "w") as json_file:
            json.dump(data_list_acc, json_file)

        arr_acc_left = np.mean(arr_acc_left); arr_acc_right = np.mean(arr_acc_right); acc = (arr_acc_left + arr_acc_right) / 2.
        print(f't_g:{t_g}; pixel_thresh:{LaneEval.pixel_thresh}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}')
        print(f'Frame[{number_frame[0]},{number_frame[1]}] - Mean_acc(Left-Right-avg): {round(arr_acc_left, 2)}%-{round(arr_acc_right, 2)}%-{round(acc, 2)}%')  
    def Save_xy_phi(seft, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        total_frame = len(json_gt); data_list = []
   
        total_time = 0.
        for i in range(0, len(json_gt)):
            gt = json_gt[i]; raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']
            mid_lane_left = gt['mid_lane_left']; mid_lane_right = gt['mid_lane_right']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue

            img = cv2.imread(raw_file_old); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            img_left = img[y_start:y_stop, (x_mid-x_ext_bot):x_mid]
            img_right = img[y_start:y_stop, x_mid:(x_mid+x_ext_bot)]

            start_time = time.time()
            # img_left = cv2.medianBlur(img_left, ksize=3)
            # img_right = cv2.medianBlur(img_right, ksize=3)
            img_left = cv2.GaussianBlur(img_left, (3, 3), 1.0)
            img_right = cv2.GaussianBlur(img_right, (3, 3), 1.0)

            arr_xy_phi_left = Voting().get_xy_phi_from_gray(img_left, flag_left=1)
            arr_xy_phi_right = Voting().get_xy_phi_from_gray(img_right, flag_left=0)

            total_point = (len(arr_xy_phi_left), len(arr_xy_phi_right))

            total_time += (time.time() - start_time)

            print(f"total_time: {round(total_time, 2)} (s) - Frame: {raw_file_new} / {total_frame} ", time.time() - start_time, "seconds") 

            data = {
                "raw_file_new": raw_file_new,
                "raw_file_old": raw_file_old,
                "lane_left": arr_xy_phi_left,
                "lane_right": arr_xy_phi_right,
                "total_point": total_point,
            }
            data_list.append(data)

        with open(f"json/Save_XY_Phi/{name}.json", "w") as json_file:
            json.dump(data_list, json_file)
        return total_time 
    def read_XY_Phi_do_HT(seft, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        json_XY_Phi = [json.loads(line) for line in open(f'json/Save_XY_Phi/{name}.json')][0]
        total_frame = len(json_gt); data_list_acc = []

        json_XY_Phi = {l['raw_file_new']: l for l in json_XY_Phi}
        arr_acc_left = []; arr_acc_right = []
        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']
            mid_lane_left = gt['mid_lane_left']; mid_lane_right = gt['mid_lane_right']
            multi_lane_left = gt['multi_lane_left']; multi_lane_right = gt['multi_lane_right']; lanes = gt['lanes']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue

            XY_Phi = json_XY_Phi[raw_file_new]
            arr_xy_phi_left = XY_Phi['lane_left']; arr_xy_phi_right = XY_Phi['lane_right']

            voting_left = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_left:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=1)
                            if Rho_vote != -1:
                                voting_left[int(Rho_vote), int(phi_vote)] += 1

            voting_right = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_right:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=0)
                            if Rho_vote != -1:
                                voting_right[int(Rho_vote), int(phi_vote)] += 1

            voting_left, voting_right = Voting().voting_factor_Rho(voting_left, voting_right, factor_Rho=8)  

            rho_xoa = 20
            voting_left[0:rho_xoa+1,] = 0 ; voting_right[0:rho_xoa+1,] = 0 # voi Roi h[320->720] 

            Rho_phi_left = Peak_Rho_phi().peak_Rho_phi_01(voting_left)
            Rho_phi_right = Peak_Rho_phi().peak_Rho_phi_01(voting_right)

            mid_lane_left = [[x,y] for x,y in mid_lane_left if y <= y_stop and y >= y_start] 
            mid_lane_right = [[x,y] for x,y in mid_lane_right if y <= y_stop and y >= y_start] 
            
            pred_lane_left = Voting().get_arr_coordiante(Rho_phi_left, flag_left=1, mid_lane_true=mid_lane_left)
            pred_lane_right = Voting().get_arr_coordiante(Rho_phi_right, flag_left=0, mid_lane_true=mid_lane_right)

            angles_left = LaneEval.get_angle(mid_lane_left)
            angles_right = LaneEval.get_angle(mid_lane_right)

            left_thresh = LaneEval.pixel_thresh / np.cos(angles_left)
            right_thresh = LaneEval.pixel_thresh / np.cos(angles_right)

            n_diem_dung_L, acc_L = LaneEval.sum_point_correct(pred_lane_left, mid_lane_left, left_thresh)

            n_diem_dung_R, acc_R = LaneEval.sum_point_correct(pred_lane_right, mid_lane_right, right_thresh)

            arr_acc_left.append(acc_L)
            arr_acc_right.append(acc_R)
            print(f'{raw_file_new} - left: {n_diem_dung_L}/{len(mid_lane_left)} - {round(acc_L, 2)}, right: {n_diem_dung_R}/{len(mid_lane_right)} - {round(acc_R, 2)} s/frame: {round(time.time() - start_time,3)}')  
            
            data_acc = {
                "raw_file_old": raw_file_old,
                "raw_file_new": raw_file_new,
                "acc_left": acc_L,
                "ratio_left": f'{n_diem_dung_L}/{len(mid_lane_left)}',
                "acc_right": acc_R,
                "ratio_right": f'{n_diem_dung_R}/{len(mid_lane_right)}',
                "Rho_phi_left": Rho_phi_left,
                "Rho_phi_right": Rho_phi_right,
                "pred_lane_left": pred_lane_left,
                "pred_lane_right": pred_lane_right,
            }
            data_list_acc.append(data_acc)

        with open(f"json/Acc/{name}_Frame[{number_frame[0]},{number_frame[1]}].json", "w") as json_file:
            json.dump(data_list_acc, json_file)

        arr_acc_left = np.mean(arr_acc_left); arr_acc_right = np.mean(arr_acc_right); acc = (arr_acc_left + arr_acc_right) / 2.
        print(f't_g:{t_g}; pixel_thresh:{LaneEval.pixel_thresh}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}')
        print(f'Frame[{number_frame[0]},{number_frame[1]}] - Mean_acc(Left-Right-avg): {round(arr_acc_left, 2)}%-{round(arr_acc_right, 2)}%-{round(acc, 2)}%')  
#############################################################################################################
def save_csv(img, path):
    img = img.astype(np.int32)
    for i in range(1,phi_max_ram+1,1):
        img[0,i] = i
    np.savetxt(path, img, delimiter=',', fmt='%d')

###########################################################333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

class Debug:
    def get_full_pre_and_post_processing(self, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        total_frame = len(json_gt); data_list = []; data_list_acc = []; data_list_rp_xy_lane = []
   
        arr_acc_left = []; arr_acc_right = []; total_time = []
        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']
            mid_lane_left = gt['mid_lane_left']; mid_lane_right = gt['mid_lane_right']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue

            img = cv2.imread(raw_file_old); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            img_left = img[y_start:y_stop, (x_mid-w_roi):x_mid]
            img_right = img[y_start:y_stop, x_mid:(x_mid+w_roi)]

            start_time = time.time()
            #img_left = cv2.medianBlur(img_left, ksize=5)
            #img_right = cv2.medianBlur(img_right, ksize=5)
            img_left = cv2.GaussianBlur(img_left, (3, 3), 0)
            img_right = cv2.GaussianBlur(img_right, (3, 3), 0)

            arr_xy_phi_left = Voting().get_xy_phi_from_gray(img_left, flag_left=1)
            arr_xy_phi_right = Voting().get_xy_phi_from_gray(img_right, flag_left=0)

            total_point = (len(arr_xy_phi_left), len(arr_xy_phi_right))

            voting_left = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_left:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=1)
                            if Rho_vote != -1:
                                voting_left[int(Rho_vote), int(phi_vote)] += 1

            voting_right = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_right:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=0)
                            if Rho_vote != -1:
                                voting_right[int(Rho_vote), int(phi_vote)] += 1

            total_time.append(time.time() - start_time)

            voting_left, voting_right = Voting().voting_factor_Rho(voting_left, voting_right, factor_Rho=8)  
            #voting_left, voting_right = Voting().voting_plus_around(voting_left, voting_right, ext_phi = 2, ext_r = 2)

            #voting_left[0:18,] = 0 ; voting_right[0:18,] = 0 # voi Roi h[290->490] w[x_mid-400,x_mid] x_top = 400/2 

            # save_csv(path=f'csv/debug/voting/{name}_{raw_file_new[-8:-4]}_left.csv', img=voting_left)
            # save_csv(path=f'csv/debug/voting/{name}_{raw_file_new[-8:-4]}_right.csv', img=voting_right)

            Rho_phi_left = Peak_Rho_phi().peak_Rho_phi_01(voting_left)
            Rho_phi_right = Peak_Rho_phi().peak_Rho_phi_01(voting_right)

            pred_lane_left = Voting().get_arr_coordiante(Rho_phi_left, flag_left=1, mid_lane_true=mid_lane_left)
            pred_lane_right = Voting().get_arr_coordiante(Rho_phi_right, flag_left=0, mid_lane_true=mid_lane_right)

            mid_lane_left = [[x,y] for x,y in mid_lane_left if y <= y_stop] 
            mid_lane_right = [[x,y] for x,y in mid_lane_right if y <= y_stop] 
            pred_lane_left = [[x,y] for x,y in pred_lane_left if y <= y_stop] 
            pred_lane_right = [[x,y] for x,y in pred_lane_right if y <= y_stop] 

            angles_left = LaneEval.get_angle(mid_lane_left)
            angles_right = LaneEval.get_angle(mid_lane_right)

            left_thresh = LaneEval.pixel_thresh / np.cos(angles_left)
            right_thresh = LaneEval.pixel_thresh / np.cos(angles_right)

            n_diem_dung_L, acc_L = LaneEval.sum_point_correct(pred_lane_left, mid_lane_left, left_thresh)

            n_diem_dung_R, acc_R = LaneEval.sum_point_correct(pred_lane_right, mid_lane_right, right_thresh)

            img = cv2.imread(raw_file_old)

            # cv2.polylines(img, [roi_left_draw], isClosed=True, color=(127,255,0), thickness=2)
            # cv2.polylines(img, [roi_right_draw], isClosed=True, color=(255,127,0), thickness=2)
            # for pt in mid_lane_left:
            #     x,y = pt[0], pt[1]
            #     cv2.circle(img, (x,y), radius=2, color=(0, 255, ), thickness=-1)
            #     cv2.line(img, (round(x-left_thresh), y), (round(x+left_thresh), y), (127,127,0), 1)
            # for pt in mid_lane_right:
            #     x,y = pt[0], pt[1]
            #     cv2.circle(img, (x,y), radius=2, color=(0, 255, ), thickness=-1)
            #     cv2.line(img, (round(x-right_thresh), y), (round(x+right_thresh), y), (127,127,0), 1)

            # for pt in pred_lane_left:
            #     cv2.circle(img, pt, radius=4, color=(0, 0, 255), thickness=1)
            # for pt in pred_lane_right:
            #     cv2.circle(img, pt, radius=4, color=(0, 0, 255), thickness=1)

            # x, y, w, h = 460-20, 240-20, 700, (310+5) - (240-20)
            # mask = np.zeros_like(img)
            # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # img = np.where(mask == 255, (255, 255, 255), img)

            # str_tmp = f't_g:{t_g}; pixel_thresh:{LaneEval.pixel_thresh}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}'
            # cv2.putText(img, str_tmp, (460,240), font, font_scale, font_color, font_thickness)
            # cv2.putText(img, f'left: {n_diem_dung_L}/{len(mid_lane_left)}, right: {n_diem_dung_R}/{len(mid_lane_right)}', (460,265), font, font_scale, font_color, font_thickness)
            # cv2.putText(img, f'acc_left: {round(acc_L, 2)}, acc_right: {round(acc_R, 2)}', (460,285), font, font_scale, font_color, font_thickness)

            arr_acc_left.append(acc_L)
            
            arr_acc_right.append(acc_R)

            # img = img[240-20:y_stop+100, x_mid-x_ext_bot-50:x_mid+x_ext_bot+50, ]
            # cv2.imwrite(f'dataset_phan_loai/{name}/{raw_file_new[-8:]}', img)

            print(f'{raw_file_new} - left: {n_diem_dung_L}/{len(mid_lane_left)} - {round(acc_L, 2)}, right: {n_diem_dung_R}/{len(mid_lane_right)} - {round(acc_R, 2)} s/frame: {round(time.time() - start_time,3)}')  
            
            data_acc = {
                "raw_file_old": raw_file_old,
                "raw_file_new": raw_file_new,
                "acc_left": acc_L,
                "ratio_left": f'{n_diem_dung_L}/{len(mid_lane_left)}',
                "acc_right": acc_R,
                "ratio_right": f'{n_diem_dung_R}/{len(mid_lane_right)}',
                "Rho_phi_left": Rho_phi_left,
                "Rho_phi_right": Rho_phi_right,
                "pred_lane_left": pred_lane_left,
                "pred_lane_right": pred_lane_right,
            }
            data_list_acc.append(data_acc)

            data = {
                "raw_file_old": raw_file_old,
                "raw_file_new": raw_file_new,
                "arr_xy_phi_left": arr_xy_phi_left,
                "arr_xy_phi_right": arr_xy_phi_right,
                "total_point": total_point,
            }
            data_list.append(data)


        with open(f"json/Save_XY_Phi/{name}_Frame[{number_frame[0]},{number_frame[1]}].json", "w") as json_file:
            json.dump(data_list, json_file)

        with open(f"json/Acc/{name}_Frame[{number_frame[0]},{number_frame[1]}].json", "w") as json_file:
            json.dump(data_list_acc, json_file)

        arr_acc_left = np.mean(arr_acc_left); arr_acc_right = np.mean(arr_acc_right); acc = (arr_acc_left + arr_acc_right) / 2.
        print(f't_g:{t_g}; pixel_thresh:{LaneEval.pixel_thresh}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}')
        print(f'Frame[{number_frame[0]},{number_frame[1]}] - Mean_acc(Left-Right-avg): {round(arr_acc_left, 2)}%-{round(arr_acc_right, 2)}%-{round(acc, 2)}%')  
        print(f'Frame[{number_frame[0]},{number_frame[1]}] - ms/frame: {round(np.mean(total_time) * 1000, 2)}') 
    def debug_full_one_image(self, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        total_frame = len(json_gt); data_list = []; data_list_acc = []; data_list_rp_xy_lane = []

        # gts = {l['raw_file_new']: l for l in json_gt}
        # json_no_lane_left = [json.loads(line) for line in open(f'json/true/{name}_no_lane_left.json').readlines()][0]
        # json_no_lane_right = [json.loads(line) for line in open(f'json/true/{name}_no_lane_right.json').readlines()][0]

        # json_no_lane_left = {l['raw_file_new']: l for l in json_no_lane_left}
        # json_no_lane_right = {l['raw_file_new']: l for l in json_no_lane_right}

        # json_blurred_lane_left = [json.loads(line) for line in open(f'json/true/{name}_blurred_lane_left.json').readlines()][0]
        # json_blurred_lane_right = [json.loads(line) for line in open(f'json/true/{name}_blurred_lane_right.json').readlines()][0]

        # json_blurred_lane_left = {l['raw_file_new']: l for l in json_blurred_lane_left}
        # json_blurred_lane_right = {l['raw_file_new']: l for l in json_blurred_lane_right}

        arr_acc_left = []; arr_acc_right = []; total_time = []
        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']
            mid_lane_left = gt['mid_lane_left']; mid_lane_right = gt['mid_lane_right']
            multi_lane_left = gt['multi_lane_left']; multi_lane_right = gt['multi_lane_right']; 
            lanes = gt['lanes']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue
            #if int(raw_file_new[-8:-4]) != number_frame: continue

            img = cv2.imread(raw_file_old); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            img_left = img[y_start:y_stop, (x_mid-w_roi):x_mid]
            img_right = img[y_start:y_stop, x_mid:(x_mid+w_roi)]

            start_time = time.time()
            #img_left = cv2.medianBlur(img_left, ksize=5)
            #img_right = cv2.medianBlur(img_right, ksize=5)
            img_left = cv2.GaussianBlur(img_left, (3, 3), 1.0)
            img_right = cv2.GaussianBlur(img_right, (3, 3), 1.0)

            arr_xy_phi_left = Voting().get_xy_phi_from_gray(img_left, flag_left=1)
            arr_xy_phi_right = Voting().get_xy_phi_from_gray(img_right, flag_left=0)

            total_point = (len(arr_xy_phi_left), len(arr_xy_phi_right))

            voting_left = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_left:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=1)
                            if Rho_vote != -1:
                                voting_left[int(Rho_vote), int(phi_vote)] += 1

            voting_right = np.zeros(shape=(Rho_max_ram+1,phi_max_ram+1), dtype=np.uint16)
            for lane in arr_xy_phi_right:
                x = lane[0]; y = lane[1]; phi = lane[2]
                for phi_tmp in range(int(phi)-phi_rotation, int(phi)+phi_rotation+1):
                    if phi_tmp >= phi_min and phi_tmp <= phi_max:
                        for x_tmp in range(-X_rotate, X_rotate+1):
                            Rho_vote, phi_vote = Voting().check_phi_doRho(phi_tmp, y, x+x_tmp, flag_left=0)
                            if Rho_vote != -1:
                                voting_right[int(Rho_vote), int(phi_vote)] += 1

            total_time.append(time.time() - start_time)

            voting_left, voting_right = Voting().voting_factor_Rho(voting_left, voting_right, factor_Rho=8)    
            rho_xoa = 0
            voting_left[0:rho_xoa+1,] = 0 ; voting_right[0:rho_xoa+1,] = 0

            # save_csv(path=f'csv/Voting/Debug/{name}_{raw_file_new[-8:-4]}_left_{Flow}_F20.csv', img=voting_left)
            # save_csv(path=f'csv/Voting/Debug/{name}_{raw_file_new[-8:-4]}_right_{Flow}.csv', img=voting_right)

            Rho_phi_left = Peak_Rho_phi().peak_Rho_phi_01(voting_left)
            Rho_phi_right = Peak_Rho_phi().peak_Rho_phi_01(voting_right)

            pred_lane_left = Voting().get_arr_coordiante(Rho_phi_left, flag_left=1, mid_lane_true=mid_lane_left)
            pred_lane_right = Voting().get_arr_coordiante(Rho_phi_right, flag_left=0, mid_lane_true=mid_lane_right)

            mid_lane_left = [[x,y] for x,y in mid_lane_left if y <= y_stop] 
            mid_lane_right = [[x,y] for x,y in mid_lane_right if y <= y_stop] 
            pred_lane_left = [[x,y] for x,y in pred_lane_left if y <= y_stop] 
            pred_lane_right = [[x,y] for x,y in pred_lane_right if y <= y_stop] 

            angles_left = LaneEval.get_angle(mid_lane_left)
            angles_right = LaneEval.get_angle(mid_lane_right)

            left_thresh = LaneEval.pixel_thresh / np.cos(angles_left)
            right_thresh = LaneEval.pixel_thresh / np.cos(angles_right)

            n_diem_dung_L, acc_L = LaneEval.sum_point_correct(pred_lane_left, mid_lane_left, left_thresh)

            n_diem_dung_R, acc_R = LaneEval.sum_point_correct(pred_lane_right, mid_lane_right, right_thresh)

            img = cv2.imread(raw_file_old)

            cv2.polylines(img, [roi_left_draw], isClosed=True, color=(127,255,0), thickness=2)
            cv2.polylines(img, [roi_right_draw], isClosed=True, color=(255,127,0), thickness=2)

            for l in lanes:
                for pt in l:
                    x,y = pt[0], pt[1]
                    cv2.circle(img, (x,y), radius=3, color=(255, 127, ), thickness=1)

            for pt in mid_lane_left:
                x,y = pt[0], pt[1]
                cv2.circle(img, (x,y), radius=2, color=(0, 255, ), thickness=-1)
                #cv2.line(img, (round(x-left_thresh), y), (round(x+left_thresh), y), (127,127,0), 1)
            for pt in mid_lane_right:
                x,y = pt[0], pt[1]
                cv2.circle(img, (x,y), radius=2, color=(0, 255, ), thickness=-1)
                #cv2.line(img, (round(x-right_thresh), y), (round(x+right_thresh), y), (127,127,0), 1)

            angles_left = LaneEval.get_angle(pred_lane_left)
            angles_right = LaneEval.get_angle(pred_lane_right)

            left_thresh = pixel_thresh_width_2dege / np.cos(angles_left)
            right_thresh = pixel_thresh_width_2dege / np.cos(angles_right)

            for pt in pred_lane_left:
                x,y = pt[0], pt[1]
                cv2.circle(img, (x,y), radius=4, color=(0, 0, 255), thickness=1)
                cv2.line(img, (round(x-left_thresh), y), (round(x+left_thresh), y), (127,127,255), 1)
            for pt in pred_lane_right:
                x,y = pt[0], pt[1]
                cv2.circle(img, (x,y), radius=4, color=(0, 0, 255), thickness=1)
                cv2.line(img, (round(x-right_thresh), y), (round(x+right_thresh), y), (127,127,255), 1)

            # x, y, w, h = 460-20, 240-20, 700, (310+5) - (240-20)
            # mask = np.zeros_like(img)
            # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # img = np.where(mask == 255, (255, 255, 255), img)

            # str_tmp = f't_g:{t_g}; pixel_thresh:{LaneEval.pixel_thresh}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}'
            # cv2.putText(img, str_tmp, (460,240), font, font_scale, font_color, font_thickness)
            cv2.putText(img, f'left: {n_diem_dung_L}/{len(mid_lane_left)}, right: {n_diem_dung_R}/{len(mid_lane_right)}', (460,265), font, font_scale, font_color, font_thickness)
            cv2.putText(img, f'acc_left: {round(acc_L, 2)}, acc_right: {round(acc_R, 2)}', (460,285), font, font_scale, font_color, font_thickness)

            arr_acc_left.append(acc_L)
            
            arr_acc_right.append(acc_R)

            img = img[240-20:y_stop+100, x_mid-x_ext_bot-50:x_mid+x_ext_bot+50, :]
            cv2.imwrite(f'my_frame_processing/debug/{name}_{raw_file_new[-8:]}', img)

            print(f'{raw_file_new} - left: {n_diem_dung_L}/{len(mid_lane_left)} - {round(acc_L, 2)}, right: {n_diem_dung_R}/{len(mid_lane_right)} - {round(acc_R, 2)} s/frame: {round(time.time() - start_time,3)}')  

        arr_acc_left = np.mean(arr_acc_left); arr_acc_right = np.mean(arr_acc_right); acc = (arr_acc_left + arr_acc_right) / 2.
        print(f't_g:{t_g}; pixel_thresh:{LaneEval.pixel_thresh}; denta_gray_max:{denta_gray_max}; ex_bot:{x_ext_bot}; ex_top:{x_ext_top}')
        print(f'Frame[{number_frame[0]},{number_frame[1]}] - Mean_acc(Left-Right-avg): {round(arr_acc_left, 2)}%-{round(arr_acc_right, 2)}%-{round(acc, 2)}%')  
        print(f'Frame[{number_frame[0]},{number_frame[1]}] - ms/frame: {round(np.mean(total_time) * 1000, 2)}') 
    def debug_get_binary_img_edge(self, name, number_frame):
        json_gt = [json.loads(line) for line in open(f'json/true/{name}.json')][0]
        total_frame = len(json_gt); data_list = []

        for i in range(0, len(json_gt)):
            gt = json_gt[i]; start_time = time.time(); raw_file_new = gt['raw_file_new']; raw_file_old = gt['raw_file_old']

            if int(raw_file_new[-8:-4]) < number_frame[0] or int(raw_file_new[-8:-4]) > number_frame[1]: continue
            #if int(raw_file_new[-8:-4]) != number_frame: continue

            img = cv2.imread(raw_file_old); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

            img_left = img[y_start:y_stop, (x_mid-w_roi):x_mid]
            img_right = img[y_start:y_stop, x_mid:(x_mid+w_roi)]

            # img_left = cv2.medianBlur(img_left, ksize=3)
            # img_right = cv2.medianBlur(img_right, ksize=3)
            img_left = cv2.GaussianBlur(img_left, (3, 3), 0)
            img_right = cv2.GaussianBlur(img_right, (3, 3), 0)
            
            voting_left = self.sub_debug_get_binary_img_edge(img_left, flag_left=1)
            voting_right = self.sub_debug_get_binary_img_edge(img_right, flag_left=0)
            cv2.polylines(voting_left, [roi_left], isClosed=True, color=(255), thickness=2)
            cv2.polylines(voting_right, [roi_right], isClosed=True, color=(255), thickness=2)

            img_meger_left_right = np.zeros(shape=(h_roi, (w_roi*2)), dtype=np.uint8)

            img_meger_left_right[:, 0:w_roi] = voting_left
            img_meger_left_right[:, w_roi:(w_roi*2)] = voting_right
            white_pixel_count = np.count_nonzero(img_meger_left_right == 255)
            img_meger_left_right[:, w_roi] = 255
            cv2.imwrite(f'my_frame_processing/debug/{name}_{raw_file_new[-8:-4]}_binary.jpg', img_meger_left_right)
            #cv2.imwrite(f'test_set/preprocessing/Binary_img/debug/{name}/{name}_{raw_file_new[-8:-4]}_idea_solution_{Flow}_debug.jpg', img_meger_left_right)

            print(f"{raw_file_new} / {total_frame}: {round(time.time() - start_time,3)} seconds - total_point: {white_pixel_count}") 
    def sub_debug_get_binary_img_edge(self, img, flag_left):
        h, w = img.shape[0], img.shape[1]
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3); sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        img_G = np.round(np.sqrt(sobel_x**2 + sobel_y**2)); img_phi = np.round(np.degrees(np.arctan2(sobel_y, sobel_x)))

        img_G[0:5,:] = 0; img_G[h-6:h,:] = 0; img_G[:, 0:5] = 0; img_G[:, w-6:w] = 0

        img_binary = np.zeros(shape=(h,w), dtype=np.uint8)
        
        img_phi_tmp = np.zeros(shape=(h,w))
        for i in range(5, h-5):
            for j in range(5, w-5):
                if img_G[i,j] > t_g:
                    phi = img_phi[i,j]; phi = Voting().check_phi(phi,flag_left)   
                    if phi > 0:
                        img_phi_tmp[i,j] = int(phi)
                    else:
                        img_G[i,j] = 0
        img_phi = img_phi_tmp

        arr_xy_phi = []
        if flag_left:
            i = 5
            while i < img.shape[0]-5:
                j = img.shape[1]-5
                flag_rising = 0
                arr_xy_candidate = []; gray_max = 0; note_left = [0, 0, 0]
                while j >= 5:
                    if mask_left[i,j] == 0: 
                        break
                    G4 = img_G[i,j-1]; G5 = img_G[i, j]; G6 = img_G[i,j+1]
                    if G5 > G4 and G5 > G6 and G4 > t_g and G6 > t_g:
                        if (img_G[i-1,j+1] > t_g or img_G[i-1,j+2] > t_g) and (img_G[i+1,j-1] > t_g or img_G[i+1,j-2] > t_g):
                            y = i; x = j
                            if Voting().check_rising(img[y,x-2:x+3], flag_left):
                                flag_rising = 1
                                x0 = x
                            elif Voting().check_falling(img[y,x-2:x+3], flag_left) and flag_rising == 1:
                                flag_rising = 0
                                deta_x = abs(x0 - x)
                                p0 = np.mean(img_phi[y,x0-1:x0+2]); p1 = np.mean(img_phi[y,x-1:x+2]); phi = (p0 + p1) / 2.
                                thresh_denta_x = pixel_thresh_width_2dege / math.cos(math.radians(phi))
                                min_data_x = 3 / math.cos(math.radians(phi))
                                if deta_x <= thresh_denta_x:
                                    x1 = int((x + x0) / 2)
                                    mean_gray = np.mean(img[y,x1-1:x1+2])
                                    # if gray_max == 0: gray_max = mean_gray
                                    # else: gray_max = (gray_max + mean_gray) / 2.
                                    if gray_max < mean_gray: gray_max = mean_gray
                                    arr_xy_candidate.append((x, x0, phi, mean_gray))
                    j = j - 1
                for pt in arr_xy_candidate:
                    if abs(gray_max - pt[-1]) <= denta_gray_max:
                        if note_left[0] < pt[0]:
                            note_left = pt
                if gray_max != 0:
                    x, x0, phi, mean_gray = note_left
                    #arr_xy_phi.append((x, y, phi))
                    img_binary[y,x] = 255; img_binary[y,x0] = 255; 
                i = i + 1
        else:
            i = 5
            while i < img.shape[0]-5:
                j = 5
                flag_rising = 0
                arr_xy_candidate = []; gray_max = 0; note_right = [w_roi, 0, 0]
                while j < img.shape[1]-5:
                    if mask_right[i,j] == 0: 
                        break
                    G4 = img_G[i,j-1]; G5 = img_G[i, j]; G6 = img_G[i,j+1]
                    if G5 > G4 and G5 > G6 and G4 > t_g and G6 > t_g:
                        if img_G[i-1,j-1] > t_g or img_G[i-1,j-2] > t_g or img_G[i+1,j+1] > t_g or img_G[i+1,j+2] > t_g:
                            y = i; x = j
                            if Voting().check_rising(img[y,x-2:x+3], flag_left):
                                flag_rising = 1
                                x0 = x
                            elif Voting().check_falling(img[y,x-2:x+3], flag_left) and flag_rising == 1:
                                flag_rising = 0
                                deta_x = abs(x0 - x)
                                p0 = np.mean(img_phi[y,x0-1:x0+2]); p1 = np.mean(img_phi[y,x-1:x+2]); phi = (p0 + p1) / 2.
                                thresh_denta_x = pixel_thresh_width_2dege / math.cos(math.radians(phi))
                                if deta_x <= thresh_denta_x:
                                    x1 = int((x + x0) / 2)
                                    mean_gray = np.mean(img[y,x1-1:x1+2])
                                    if gray_max < mean_gray: gray_max = mean_gray
                                    arr_xy_candidate.append((x, x0, phi, mean_gray))
                    j = j + 1
                for pt in arr_xy_candidate:
                    if gray_max - pt[-1] <= denta_gray_max:
                        if note_right[0] > pt[0]:
                            note_right = pt
                if gray_max != 0:
                    x, x0, phi, mean_gray = note_right
                    #if mean_gray > 110:
                    img_binary[y,x] = 255; img_binary[y,x0] = 255; 
                i = i + 1
        return img_binary

class Peak_Rho_phi:
    def peak_Rho_phi_01(self, voting):
        max_value = np.max(voting)
        Rho_phi = []
        
        max_index = np.unravel_index(np.argmax(voting, axis=None), voting.shape)
        Rho, phi = int(max_index[0]), int(max_index[1])
        #print(f'Max_vote: {max_value} -  Rho: {Rho} -  Phi: {phi}')
        Rho_phi = [(Rho , phi)]
        return Rho_phi
   
