import os
import matplotlib.pyplot as plt
import math
import scipy
import cv2
import numpy as np

label_dir = "training/labels/"
all_files = os.listdir(label_dir)
txt_files = filter(lambda x: x[-4:] == ".txt", all_files)
max_height = -1
max_width = -1
bb_map = {}
hw_dict = {}
for fname in txt_files:
	f = open(label_dir + fname, 'r')
	bb_map[fname[:-4]] = []
	for line in f:
		line = line.split()
		left_x = float(line[4])
		left_y = float(line[5])
		right_x = float(line[6])
		right_y = float(line[7])
		width = abs(right_x - left_x)
		height = abs(right_y - left_y)
		mag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))
		if line[0] == "Car": # use this to toggle object type
			ratio = int(height/width)
			hw_dict[ratio] = hw_dict.get(ratio, 0) + 1
			if height > max_height:
				max_height = height
				max_width = width
			bb_map[fname[:-4]].append((int(left_x), int(left_y), int(right_x), int(right_y)))
	f.close()

img_dir = "training/images/"
all_files = os.listdir(img_dir)
img_files = filter(lambda x: x[-4:] == ".png", all_files)
compositex = np.zeros((int(max_height), int(max_width)))
compositey = np.zeros((int(max_height), int(max_width)))
count = 0
for img_fname in img_files:
	image = cv2.imread(img_dir+img_fname)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	map_name = img_fname[:-4]
	for bb in bb_map[map_name]:
		count += 1
		cyclist = image[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
		cyclist = cv2.resize(cyclist, (int(max_width), int(max_height)))
		x = cv2.Sobel(cyclist, cv2.CV_64F, 1, 0, ksize=5)
		y = cv2.Sobel(cyclist, cv2.CV_64F, 0, 1, ksize=5)
		compositex += x
		compositey += y
compositex = np.divide(compositex, count)
compositey = np.divide(compositey, count)
cv2.imwrite("compositex.jpg", compositex)
cv2.imwrite("compositey.jpg", compositey)
