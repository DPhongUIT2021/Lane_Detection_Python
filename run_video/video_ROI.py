import numpy as np
import cv2
import os
import math
import video_lib as Vlib

#=======path============
folder_Video = 'D:/CE201.N21_DoAn1/Software_Design/Video/BGR/Origin'
output_path = 'D:/CE201.N21_DoAn1/Software_Design/Video/BGR/640x120'


# Define the font properties for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)  # White color
thickness = 1


# Draw the text using FONT_HERSHEY_SIMPLEX


#==========convert .xlsx .cvs=========
for name in os.listdir(folder_Video):
    if name != 'video_06.mp4':
            continue
    cap = cv2.VideoCapture(folder_Video + "/" + name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    delay_between_frame = int(1000 / fps)

    out = cv2.VideoWriter(output_path + '/' + name[:-4] + "_640x120.mp4", fourcc, fps, (width, int(height/3)), isColor=True)

    while cap.isOpened():
        
        ret, img_orgin = cap.read()
        
        if not ret:
            break
        img = cv2.resize(img_orgin, (640, 360))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_roi_h_div_3 = Vlib.get_roi(img, div = 3)
        h, w = img.shape[0], img.shape[1]
        img_left = img_roi_h_div_3[:, 0:int(w/2)]
        img_right = img_roi_h_div_3[:, int(w/2):w]
        #img_roi_h_div_2 = Vlib.get_roi(img, div = 2)
        #img = img.astype(np.uint8)
        
        # text_fps = f"FPS: {int(fps)}"
        # text_frame_counter = f"Frame: {frame_count}/{total_frames}"
        frame_number += 1
        # text_resolution = f"Resolution: {width}x{height}"

        # cv2.putText(img, text_resolution, (10, 20), font, font_scale, font_color, thickness)
        # cv2.putText(img, text_fps, (10, 35), font, font_scale, font_color, thickness)
        # cv2.putText(img, text_frame_counter, (10, 50), font, font_scale, font_color, thickness)

        #out.write(img_roi_h_div_3)
        cv2.imshow('Origin', img_orgin)
        cv2.imshow('Roi/3', img_roi_h_div_3)
        cv2.imshow('Right', img_right)
        cv2.imshow('Left', img_left)
        #out.write(img_roi_h_div_3)
        #cv2.imshow(name+'roi h/3', img_roi_h_div_3)
        #cv2.imshow(name+'roi h/2', img_roi_h_div_2)

        # image_path = f"{output_path}/{frame_number:04d}.jpg"
        #cv2.imwrite(image_path, img)
        # print(frame_number)

        
        if cv2.waitKey(delay_between_frame) & 0xFF == 27:  # Press 'Esc' to exit
            break
    print("Done ", name)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

