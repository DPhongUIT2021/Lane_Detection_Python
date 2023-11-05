
import os
import lib_me as lib
import time
from lib_TU_me import LaneEval
import openpyxl
import json
import numpy as np

wb = openpyxl.Workbook()
ws = wb.active

if __name__ == '__main__':

        arr_name = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test3_shadow', 'test4_noline', 'test5_arrow', 'test6_curve', 'test7_cross', 'test8_night']
        start_time = time.time()

        LaneEval.pixel_thresh = 15
        lib.X_rotate = 2
        lib.phi_rotation = 5

        lib.denta_gray_max = 100
        lib.t_g = 10
        lib.thresh_denta_x = 30
        lib.pixel_thresh_width_2dege  = 15

        arr_total_time = []

        #=======Debug()==============================================================================================
        name = 'test2_hlight'; number_frame = [258,265]
        # #lib.Debug().get_full_pre_and_post_processing(name, number_frame)
        #lib.Debug().debug_full_one_image(name, number_frame)
        #lib.Debug().debug_get_binary_img_edge(name, number_frame)
        #lib.Analysis().analysis_voting(name='0530', number_frame=510)


        #=======lib.Voting()==============================================================================================
        name = 'test2_hlight'; number_frame = [0,485]
        #name = 'test0_normal'; number_frame = [7000,9620]
        lib.Voting().Save_xy_phi(name, number_frame)
        lib.Voting().read_XY_Phi_do_HT(name, number_frame)

        #lib.Voting().Save_voting_PKL_after_HT(name, number_frame)
        #lib.Voting().read_voting_PKL_to_end(name, number_frame)

        # name = 'test2_hlight'; number_frame = [0,485]
        # #lib.Voting().Save_voting_PKL_after_HT(name, number_frame)
        # #lib.Voting().read_voting_PKL_to_end(name, number_frame)

        # name = 'test3_shadow'; number_frame = [0,929]
        # #lib.Voting().Save_voting_PKL_after_HT(name, number_frame)
        # #lib.Voting().read_voting_PKL_to_end(name, number_frame)

        # name = 'test5_arrow'; number_frame = [0,889]
        # #lib.Voting().Save_voting_PKL_after_HT(name, number_frame)
        # #lib.Voting().read_voting_PKL_to_end(name, number_frame)

        # name = 'test8_night'; number_frame = [0,7028]
        #lib.Voting().Save_voting_PKL_after_HT(name, number_frame)
        #lib.Voting().read_voting_PKL_to_end(name, number_frame)

        #=======Peak_Rho_phi()==============================================================================================       
        #=======Analysis()==============================================================================================



        print(f"Total_Time:", time.time() - start_time, "seconds") 


       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       

        print('Done main.py')

