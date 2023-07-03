import numpy as np
import cv2
import math
import threading
import time
import multiprocessing


pi_div_2 = math.pi / 2
phi_left_min = 25
phi_left_max = 80
phi_right_min = 105
phi_right_max = 153
def get_roi(img, div):
    #Get ROI height/2
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    h_roi = int(h / div)
    h_roi = h_roi * -1
    img = img[h_roi:h, :]

    return img

phi_max = 90
r_max = 300 # imge 640x120 phi(100,170)
T = 100


def get_Rho_phi_left(img): #input: gray -> edge
    h, w = img.shape[0], img.shape[1]
    rho_img = np.zeros(shape=(h,w), dtype=np.uint8)
    phi_img = np.zeros(shape=(h,w), dtype=np.uint8)
    for i_outside in range(1, h-5, 5):
        for j_outside in range(1, w-5, 5):
            gradien_max = (0,0,0)
            check_angel = False
            for i in range(i_outside + 1, i_outside + 4): # i = [1,3]
                for j in range(j_outside + 1, j_outside + 4):
                    gx = - 1 * img[i-1, j-1] + 0 * img[i-1, j] + 1 * img[i-1, j+1] \
                            - 2 * img[i, j-1] + 0 * img[i, j] + 2 * img[i, j+1]   \
                            - 1 * img[i+1, j-1] + 0 * img[i+1, j] + 1 * img[i+1, j+1]

                    gy = - 1 * img[i-1, j-1] - 2 * img[i-1, j] - 1 * img[i-1, j+1] \
                        + 0 * img[i, j-1] + 0 * img[i, j] + 0 * img[i, j+1]   \
                        + 1 * img[i+1, j-1] + 2 * img[i+1, j] + 1 * img[i+1, j+1]
                    gradien = int(math.sqrt(gx**2 + gy**2))
                    angle = math.atan2(gx,gy)
                    phi = math.degrees(angle)
                    phi = int(phi)
                    if(phi < 0):
                        phi += 180      
                    # call module Non-Maximum Suppress
                    if phi >= phi_left_min and phi <= phi_left_max:
                        check_angel = True
                        if gradien_max[0] < gradien:
                            gradien_max = (gradien, i, j, phi)
            if gradien_max[0] >= T and check_angel:
                y, x, phi_dgree = gradien_max[1], gradien_max[2], gradien_max[3]

                phi = math.radians(phi_dgree)
                rho = x*math.sin(phi) + y*math.cos(phi)

                rho = round(rho)
                phi_img[y,x] = phi_dgree
                rho_img[y,x] = rho

    return phi_img, rho_img

def get_Rho_phi_right(img): #input: gray -> edge
    h, w = img.shape[0], img.shape[1]
    rho_img = np.zeros(shape=(h,w), dtype=np.uint8)
    phi_img = np.zeros(shape=(h,w), dtype=np.uint8)
    for i_outside in range(1, h-5, 5):
        for j_outside in range(1, w-5, 5):
            gradien_max = (0,0,0)
            check_angel = False
            for i in range(i_outside + 1, i_outside + 4): # i = [1,3]
                for j in range(j_outside + 1, j_outside + 4):
                    gx = - 1 * img[i-1, j-1] + 0 * img[i-1, j] + 1 * img[i-1, j+1] \
                            - 2 * img[i, j-1] + 0 * img[i, j] + 2 * img[i, j+1]   \
                            - 1 * img[i+1, j-1] + 0 * img[i+1, j] + 1 * img[i+1, j+1]

                    gy = - 1 * img[i-1, j-1] - 2 * img[i-1, j] - 1 * img[i-1, j+1] \
                        + 0 * img[i, j-1] + 0 * img[i, j] + 0 * img[i, j+1]   \
                        + 1 * img[i+1, j-1] + 2 * img[i+1, j] + 1 * img[i+1, j+1]
                    gradien = int(math.sqrt(gx**2 + gy**2))
                    angle = math.atan2(gx,gy)
                    phi = math.degrees(angle)
                    phi = int(phi)
                    if(phi < 0):
                        phi += 180      
                    # call module Non-Maximum Suppress
                    if phi >= phi_right_min and phi <= phi_right_max:
                        check_angel = True
                        if gradien_max[0] < gradien:
                            gradien_max = (gradien, i, j, phi)

            if gradien_max[0] >= T and check_angel:
                y, x, phi_dgree = gradien_max[1], gradien_max[2], gradien_max[3]
                phi_dgree = phi_dgree - 90

                phi = math.radians(phi_dgree)
                rho = (w-1-x)*math.cos(phi) + y*math.sin(phi)
                rho = round(rho)
               
                phi_img[y,x] = phi_dgree
                rho_img[y,x] = rho

    return phi_img, rho_img


def sobel_nms_fixT_vote_left(img): 
    h, w = img.shape[0], img.shape[1]
    voting_left = np.zeros(shape=(r_max,phi_max), dtype=np.uint16)
    for i_outside in range(1, h-5, 5):
        for j_outside in range(1, w-5, 5):
            gradien_max = (0,0,0,0)
            check_angel = False
            for i in range(i_outside + 1, i_outside + 4): # i = [1,3]
                for j in range(j_outside + 1, j_outside + 4):
                    gx = - 1 * img[i-1, j-1] + 0 * img[i-1, j] + 1 * img[i-1, j+1] \
                            - 2 * img[i, j-1] + 0 * img[i, j] + 2 * img[i, j+1]   \
                            - 1 * img[i+1, j-1] + 0 * img[i+1, j] + 1 * img[i+1, j+1]
                    gy = - 1 * img[i-1, j-1] - 2 * img[i-1, j] - 1 * img[i-1, j+1] \
                        + 0 * img[i, j-1] + 0 * img[i, j] + 0 * img[i, j+1]   \
                        + 1 * img[i+1, j-1] + 2 * img[i+1, j] + 1 * img[i+1, j+1]
                    gradien = int(math.sqrt(gx**2 + gy**2))
                    angle = math.atan2(gx,gy)
                    phi = math.degrees(angle)
                    phi = int(phi)
                    if(phi < 0):
                        phi += 180      
                    # call module Non-Maximum Suppress
                    if phi >= phi_left_min and phi <= phi_left_max:
                        check_angel = True
                        if gradien_max[0] < gradien:
                            gradien_max = (gradien, i, j, phi)

            if gradien_max[0] >= T and check_angel:
                y, x, phi_dgree = gradien_max[1], gradien_max[2], gradien_max[3]

                #phi = math.radians(phi_dgree)
                for phi_step in range(phi_dgree-2,phi_dgree+3):
                    phi = math.radians(phi_step)
                    rho = x*math.sin(phi) + y*math.cos(phi)
                    rho = round(rho)
                    if rho < r_max - 5:
                        voting_left[rho, phi_step] +=1
                # rho = x*math.sin(phi) + y*math.cos(phi)
                # rho = round(rho)
                #voting_left[rho, phi_dgree] +=1
                # if rho < r_max - 5:
                #     for rho_next in range(-5,6):
                #         for phi_next in range(-5,6):
                #             voting_left[rho+rho_next, phi_dgree+phi_next] +=1
                    # voting_left[rho, phi_dgree] = voting_left[rho-1, phi_dgree-1] + voting_left[rho-1, phi_dgree] + voting_left[rho-1, phi_dgree+1] \
                    #                             + voting_left[rho, phi_dgree-1] + voting_left[rho, phi_dgree] + voting_left[rho, phi_dgree+1]\
                    #                             + voting_left[rho+1, phi_dgree-1] + voting_left[rho+1, phi_dgree] + voting_left[rho+1, phi_dgree-1]
                    # voting_left[rho, phi_dgree] =                                 ( voting_left[rho-1, phi_dgree] +                            \
                    #                             + voting_left[rho, phi_dgree-1] + voting_left[rho, phi_dgree] + voting_left[rho, phi_dgree+1]\
                    #                             +                                + voting_left[rho+1, phi_dgree] ) 
    return voting_left

def sobel_nms_fixT_vote_right(img): 
    h, w = img.shape[0], img.shape[1]
    voting_right = np.zeros(shape=(r_max,phi_max), dtype=np.uint32)
    for i_outside in range(1, h-5, 5):
        for j_outside in range(1, w-5, 5):
            gradien_max = (0,0,0,0)
            check_angel = False
            for i in range(i_outside + 1, i_outside + 4): # i = [1,3]
                for j in range(j_outside + 1, j_outside + 4):
                    gx = - 1 * img[i-1, j-1] + 0 * img[i-1, j] + 1 * img[i-1, j+1] \
                            - 2 * img[i, j-1] + 0 * img[i, j] + 2 * img[i, j+1]   \
                            - 1 * img[i+1, j-1] + 0 * img[i+1, j] + 1 * img[i+1, j+1]
                    gy = - 1 * img[i-1, j-1] - 2 * img[i-1, j] - 1 * img[i-1, j+1] \
                        + 0 * img[i, j-1] + 0 * img[i, j] + 0 * img[i, j+1]   \
                        + 1 * img[i+1, j-1] + 2 * img[i+1, j] + 1 * img[i+1, j+1]
                    gradien = int(math.sqrt(gx**2 + gy**2))
                    angle = math.atan2(gx,gy)
                    phi = math.degrees(angle)
                    phi = int(phi)
                    if(phi < 0):
                        phi += 180      
                    # call module Non-Maximum Suppress
                    if phi >= phi_right_min and phi <= phi_right_max:
                        check_angel = True
                        if gradien_max[0] < gradien:
                            gradien_max = (gradien, i, j, phi)

            if gradien_max[0] >= T and check_angel:
                y, x, phi_dgree = gradien_max[1], gradien_max[2], gradien_max[3]
                phi_dgree = phi_dgree - 90

                for phi_step in range(phi_dgree-2,phi_dgree+3):
                    phi = math.radians(phi_step)
                    rho = (w-1-x)*math.cos(phi) + y*math.sin(phi)
                    rho = round(rho)
                    if rho < r_max - 5:
                        voting_right[rho, phi_step] +=1
                        
                # phi = math.radians(phi_dgree)
                # rho = (w-1-x)*math.cos(phi) + y*math.sin(phi)
                # rho = round(rho)
                # if rho < r_max - 5:
                #     for rho_next in range(-5,6):
                #         for phi_next in range(-5,6):
                #                 voting_right[rho+rho_next, phi_dgree+phi_next] +=1
                    # voting_right[rho, phi_dgree] = voting_right[rho-1, phi_dgree-1] + voting_right[rho-1, phi_dgree] + voting_right[rho-1, phi_dgree+1] \
                    #                             + voting_right[rho, phi_dgree-1] + voting_right[rho, phi_dgree] + voting_right[rho, phi_dgree+1]\
                    #                             + voting_right[rho+1, phi_dgree-1] + voting_right[rho+1, phi_dgree] + voting_right[rho+1, phi_dgree-1]
                    # voting_right[rho, phi_dgree] =                                 ( voting_right[rho-1, phi_dgree] +                            \
                    #                             + voting_right[rho, phi_dgree-1] + voting_right[rho, phi_dgree] + voting_right[rho, phi_dgree+1]\
                    #                             +                                + voting_right[rho+1, phi_dgree] )                   
    return voting_right  


def draw_line_left(img, voting_left):
    h, w = img.shape[0], img.shape[1]
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    left_index = np.unravel_index(voting_left.argmax(), voting_left.shape)
       
    phi_left = math.radians(left_index[1])
    r_left = left_index[0]
    sin_phi = math.sin(phi_left)
    cos_phi = math.cos(phi_left)
    y = 0
    x = r_left / sin_phi
    if x >= w:
        x = w - 1
        y = (r_left - x * sin_phi) / cos_phi
    top = (round(x), round(y))

    y = h
    x = (r_left - h * cos_phi) / sin_phi
    if x < 0:
        x = 0
        y = r_left / cos_phi
    buttom = (round(x), round(y))

    img = cv2.line(img, top, buttom, (0,0,255), 2)
    cv2.imshow("Left", img)

    return top, buttom


def draw_line_right(img, voting_right):
    h, w = img.shape[0], img.shape[1]
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    right_index = np.unravel_index(voting_right.argmax(), voting_right.shape)

    phi_right = math.radians(right_index[1])
    r_right = right_index[0]
    sin_phi = math.sin(phi_right)
    cos_phi = math.cos(phi_right)
    y = 0
    k = r_right / cos_phi
    x = (w-1) - k
    if k > (w-1):
        x = 0
        y = (r_right -(w -1 )*cos_phi) / sin_phi
    top = (round(x), round(y))

    y = h
    k = (r_right - h * sin_phi) / cos_phi
    x = (w - 1) - k
    if k < 0:
        x = w - 1
        y = r_right / sin_phi 
    buttom = (round(x), round(y))

    img = cv2.line(img, top, buttom, (0,0,255), 2)
    cv2.imshow("Right", img)

    return top, buttom
    

def sobel_nms_fixT_get_binary_img_right(img): 
    h, w = img.shape[0], img.shape[1]
    img_new = np.zeros_like(img)
    for i_outside in range(1, h-5, 5):
        for j_outside in range(1, w-5, 5):
            gradien_max = (0,0,0,0)
            check_angel = False
            for i in range(i_outside + 1, i_outside + 4): # i = [1,3]
                for j in range(j_outside + 1, j_outside + 4):
                    gx = - 1 * img[i-1, j-1] + 0 * img[i-1, j] + 1 * img[i-1, j+1] \
                            - 2 * img[i, j-1] + 0 * img[i, j] + 2 * img[i, j+1]   \
                            - 1 * img[i+1, j-1] + 0 * img[i+1, j] + 1 * img[i+1, j+1]
                    gy = - 1 * img[i-1, j-1] - 2 * img[i-1, j] - 1 * img[i-1, j+1] \
                        + 0 * img[i, j-1] + 0 * img[i, j] + 0 * img[i, j+1]   \
                        + 1 * img[i+1, j-1] + 2 * img[i+1, j] + 1 * img[i+1, j+1]
                    gradien = int(math.sqrt(gx**2 + gy**2))
                    angle = math.atan2(gx,gy)
                    phi = math.degrees(angle)
                    phi = int(phi)
                    if(phi < 0):
                        phi += 180      
                    # call module Non-Maximum Suppress
                    if phi >= phi_right_min and phi <= phi_right_max:
                        check_angel = True
                        if gradien_max[0] < gradien:
                            gradien_max = (gradien, i, j, phi)

            if gradien_max[0] >= T and check_angel:
                y, x = gradien_max[1], gradien_max[2]
                img_new[y, x] = 255

    return img_new


def sobel_nms_fixT_get_binary_img_left(img): 
    h, w = img.shape[0], img.shape[1]
    img_new = np.zeros_like(img)
    for i_outside in range(1, h-5, 5):
        for j_outside in range(1, w-5, 5):
            gradien_max = (0,0,0,0)
            check_angel = False
            for i in range(i_outside + 1, i_outside + 4): # i = [1,3]
                for j in range(j_outside + 1, j_outside + 4):
                    gx = - 1 * img[i-1, j-1] + 0 * img[i-1, j] + 1 * img[i-1, j+1] \
                            - 2 * img[i, j-1] + 0 * img[i, j] + 2 * img[i, j+1]   \
                            - 1 * img[i+1, j-1] + 0 * img[i+1, j] + 1 * img[i+1, j+1]
                    gy = - 1 * img[i-1, j-1] - 2 * img[i-1, j] - 1 * img[i-1, j+1] \
                        + 0 * img[i, j-1] + 0 * img[i, j] + 0 * img[i, j+1]   \
                        + 1 * img[i+1, j-1] + 2 * img[i+1, j] + 1 * img[i+1, j+1]
                    gradien = int(math.sqrt(gx**2 + gy**2))
                    angle = math.atan2(gx,gy)
                    phi = math.degrees(angle)
                    phi = int(phi)
                    if(phi < 0):
                        phi += 180      
                    # call module Non-Maximum Suppress
                    if phi >= phi_left_min and phi <= phi_left_max:
                        check_angel = True
                        if gradien_max[0] < gradien:
                            gradien_max = (gradien, i, j, phi)

                if gradien_max[0] >= T and check_angel:
                    y, x = gradien_max[1], gradien_max[2]
                    img_new[y, x] = 255

    return img_new










