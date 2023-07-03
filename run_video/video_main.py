import cv2
import os
import time
import video_lib as Vlib
import multiprocessing
import numpy as np
# import openpyxl
import tmp as tmp
#=======path============
folder_Video = '/media/dell/DATA_UIT/CE201.N21_DoAn1/Software_Design/Video/BGR/640x120'
folder_out_Vote = 'D:/CE201.N21_DoAn1/Software_Design/Video/Vote/video_07'
folder_out_Rho_phi = 'D:/CE201.N21_DoAn1/Software_Design/Video/Rho_phi/video_07'
folder_Analysis = 'D:/CE201.N21_DoAn1/Software_Design/Video/Analysis'


# Define the font properties for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)  # White color
thickness = 1

# wb = openpyxl.Workbook()
# # select the active worksheet
# ws = wb.active

def main():
    for name in os.listdir(folder_Video):
        if name != 'video_07_640x120.mp4':
            continue
        cap = cv2.VideoCapture(folder_Video + "/" + name)
        print(name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        w_div = int(width/2)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        #delay_between_frame = int(1000 / fps)
        row = 3

        text_fps = f"FPS: {int(fps)}"
        text_resolution = f"ROI: {width}x{height}"

        while cap.isOpened():
            
            ret, img_bgr = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_left = img[:, 0:w_div]
            img_right = img[:, w_div:width]

            start_time = time.time()
            # img_right = Vlib.sobel_nms_fixT_get_binary_img_right(img_right)
            # img_left = Vlib.sobel_nms_fixT_get_binary_img_left(img_left)
            

            voting_left = Vlib.sobel_nms_fixT_vote_left(img_left)
            voting_right = Vlib.sobel_nms_fixT_vote_right(img_right)
            #tmp.save_Rho_phi_Vote_left_Vote_right(name, voting_left, voting_right, frame_count, folder_out_Vote)
            
            top_left, buttom_left = Vlib.draw_line_left(img_left, voting_left)
            top_right, buttom_right = Vlib.draw_line_right(img_right, voting_right)
            execution_time = time.time() - start_time; 
            text_frame_counter = f"Frame: {frame_count}/{total_frames}"
            frame_count += 1
            cv2.putText(img_bgr, text_resolution, (10, 20), font, font_scale, font_color, thickness)
            cv2.putText(img_bgr, text_frame_counter, (10, 35), font, font_scale, font_color, thickness)
            cv2.putText(img_bgr, f"Time execution 1 frame: {execution_time} second", (10, 50), font, font_scale, font_color, thickness)

            cv2.imshow(name, img_bgr)
             
            print(f"Time of frame : {frame_count}: {execution_time} giây")

            #print('Frame: ' ,frame_count - 1)
            # left_index = np.unravel_index(voting_left.argmax(), voting_left.shape)
            # right_index = np.unravel_index(voting_right.argmax(), voting_right.shape) 
            # print('voting max left:', voting_left.max(), 'left_index:(r,phi)=', left_index)
            # print('voting max right:', voting_right.max(), 'right_index:(r,phi)=', right_index)

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break
    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #multiprocessing.freeze_support()
    main()
    #wb.save(folder_Analysis + '/video_06_right.xlsx')



