import cv2
import numpy as np
import os
from distutils.dir_util import copy_tree

# For Mouse Event parameters
class MouseParams:
    def __init__(self, img_src, img_dst, coordinate_path):
        self.img_src = img_src
        self.img_dst = img_dst
        self.src_img_point = []
        self.dst_img_point = []
        self.coordinate_path = coordinate_path
        self.order = ""

# Mouse events (save / Delete)
def onMouse(event, x, y, flags, param) :
    global drawing
    mp = param
    
    if event == cv2.EVENT_LBUTTONDOWN :
        print('Coordinate of the target dot: ', x, y)
        if mp.order == 'front':
            # Dot
            drawing = True

            # Source Image Coordinate
            cv2.circle(mp.img_src,(x,y),8,(0,0,255),-1)
            mp.src_img_point.append(x)
            mp.src_img_point.append(y)

            # For Debug Coordinate txt file
            f = open(f"{mp.coordinate_path}/HP_coordinate.txt", "a")
            data = str(x) + ", " + str(y) + "\n"
            f.write(data)
            f.close()
            
        if mp.order == 'back':
            # Dot
            drawing = True

            # Target Image Coordinate
            cv2.circle(mp.img_dst,(x,y),8,(0,0,255),-1)
            mp.dst_img_point.append(x)
            mp.dst_img_point.append(y)

            # For Debug Coordinate txt file
            f = open(f"{mp.coordinate_path}/LP_coordinate.txt", "a")
            data = str(x) + ", " + str(y)  + "\n"
            f.write(data)
            f.close()

# Homography Transformation
def Homography_warp(base_folder, lr_plate_location, hr_plate_location):
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            result_path = f"{folder_path}/Homography_Transformation"
            coordinate_path = f"{result_path}/coordinate"
            output_path = os.path.join('output', folder_name)
            lr_plate_path = os.path.join(folder_path, lr_plate_location)
            hr_plate_path = os.path.join(folder_path, hr_plate_location)
            front_flag = 0

            os.makedirs(coordinate_path,  exist_ok = True)

            print(f"\n현재 폴더: {folder_path}")

            # TXT File for coordinate
            f = open(f"{coordinate_path}/HP_coordinate.txt", "w", encoding="UTF-8")
            f.close()
            
            while True:
                if front_flag != 1:
                    img_src = cv2.imread(f'{hr_plate_path}/HR.png')
                    img_cpy = cv2.imread(f'{hr_plate_path}/HR.png')
                    img_dst = cv2.imread(f'{lr_plate_path}/16.png')
                    img_dst_cpy = cv2.imread(f'{lr_plate_path}/16.png')
                    
                    # For easy to click (No x4 Resolution image is too small)
                    img_src = cv2.resize(img_src, (img_src.shape[1]*4, img_src.shape[0]*4), interpolation=cv2.INTER_CUBIC)
                    img_cpy = cv2.resize(img_cpy, (img_cpy.shape[1]*4, img_cpy.shape[0]*4), interpolation=cv2.INTER_CUBIC)
                    img_dst = cv2.resize(img_dst, (img_src.shape[1], img_src.shape[0]), interpolation=cv2.INTER_CUBIC)
                    img_dst_cpy = cv2.resize(img_dst_cpy, (img_cpy.shape[1], img_cpy.shape[0]), interpolation=cv2.INTER_CUBIC)
                    
                    mp = MouseParams(img_src, img_dst, coordinate_path)
                    mp.order = "front"
                   
                    cv2.imshow('SRC', img_src)
                    cv2.setMouseCallback('SRC', onMouse, mp)

                    key = cv2.waitKey()
                    
                    if key == ord("c"):
                        mp.src_img_point.clear()
                        print("Remove src_img_point Coordinates!")

                    else:
                        cv2.destroyAllWindows()
                        front_flag = 1

                if front_flag == 1:
                    mp.order = "back"

                    cv2.imshow('DST', img_dst)
                    cv2.setMouseCallback('DST', onMouse, mp)
                    
                    key = cv2.waitKey()

                    if key == ord("c"):
                        mp.dst_img_point.clear()
                        print("Remove dst_img_point Coordinates!")

                    else:
                        cv2.destroyAllWindows()
                        break
                
            
            array_src = np.array(mp.src_img_point)
            point_src = array_src.reshape(4,2)
            array_dst = np.array(mp.dst_img_point)
            point_dst = array_dst.reshape(4,2)

            f = open(f"{coordinate_path}/LP_coordinate.txt", "w",  encoding="UTF-8")
            f.close()
     
            h, trash_val = cv2.findHomography(point_src, point_dst)
            
            warp_check = cv2.warpPerspective(img_src, h, (img_src.shape[1], img_src.shape[0]))
            img_output = cv2.warpPerspective(img_cpy, h, (img_src.shape[1], img_src.shape[0]))

            warp_check = cv2.resize(warp_check, (warp_check.shape[1]//4, warp_check.shape[0]//4), interpolation=cv2.INTER_AREA)
            img_output = cv2.resize(img_output, (img_output.shape[1]//4, img_output.shape[0]//4), interpolation=cv2.INTER_AREA)

            
            cv2.imwrite(f'{output_path}/Pseudo_GT.png', img_output)

            cv2.waitKey()



def initial_settings(base_folder, lr_plate_path):
    for folder_name in os.listdir(base_folder):

        # Must be modified folder name
        result_path = os.path.join('output', folder_name, 'Plate_crop')
        plate_path = os.path.join(base_folder, folder_name, lr_plate_path)
        
        os.makedirs(result_path,  exist_ok = True)
        copy_tree(plate_path, result_path)