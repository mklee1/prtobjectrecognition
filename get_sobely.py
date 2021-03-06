import os
import matplotlib.pyplot as plt
import math
import scipy
import cv2
import numpy as np

def contrast(src, dest):
    comp = cv2.imread(src)

    flat_cy = comp.flatten()
    avg = sum(flat_cy)/len(flat_cy)
    avg = 60
    print (avg)

    for i in range(len(comp)):
        for j in range(len(comp[0])):
            px = comp[i,j]
            if px[0] > avg:
                comp[i,j] = [255, 255, 255]
            else:
                comp[i,j] = [0, 0, 0]

    cv2.imwrite(dest, comp)

def outline():
    label_dir = "testing/labels/"
    all_files = os.listdir(label_dir)
    txt_files = filter(lambda x: x[-4:] == ".txt", all_files)
    bb_map = {}
    hw_dict = {}
    actual_objs = []
    car_mask = cv2.imread("mask_car.jpg")
    ped_mask = cv2.imread("mask_pedestrian.jpg")
    cyc_mask = cv2.imread("mask_cyclist.jpg")
    count = 0
    dontcare = 0
    for fname in txt_files:
        f = open(label_dir + fname, 'r')
        bb_map[fname[:-4]] = []
        # get bounding box info
        for line in f:
            count += 1
            line = line.split()
            left_x = float(line[4])
            left_y = float(line[5])
            right_x = float(line[6])
            right_y = float(line[7])
            bb_map[fname[:-4]].append((int(left_x), int(left_y), int(right_x), int(right_y)))
            actual_objs.append(line[0])
            if (line[0] == 'DontCare'): 
                dontcare += 1
        f.close()

    img_dir = "testing/images/"
    all_files = os.listdir(img_dir)
    img_files = filter(lambda x: x[-4:] == "00385*.png", all_files)
    result = ""
    predicted_objs = ""
    count = 0
    success = 0
    for img_fname in ['003878.png']:
        print (img_fname)
        image = cv2.imread(img_dir+img_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        map_name = img_fname[:-4]
        if int(map_name) > (3470+500): # only do 500 images
            break
        for bb in bb_map[map_name]:
            count += 1
            obj = image[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
            obj_width = len(obj)
            obj_height = len(obj[0])

            mask_car_new = cv2.resize(car_mask, (int(obj_width), int(obj_height)))
            mask_ped_new = cv2.resize(ped_mask, (int(obj_width), int(obj_height)))
            mask_cyc_new = cv2.resize(cyc_mask, (int(obj_width), int(obj_height)))

            obj_sobel = cv2.Sobel(obj, cv2.CV_64F, 0, 1, ksize=5)
            # compare mask to Sobel Y image
            
            currfile = 'img' + str(count) + '.jpg'

            for i in range(len(obj_sobel)):
                for j in range(len(obj_sobel[0])):
                    px = obj_sobel[i,j]
                    if px > 60:
                        obj_sobel[i,j] = 255
                    else:
                        obj_sobel[i,j] = 0  
            cv2.imwrite(currfile, obj_sobel)
outline()