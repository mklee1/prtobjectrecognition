import os
import matplotlib.pyplot as plt
import math
import scipy
import cv2
from sklearn import svm

all_files = os.listdir("training/labels/")
txt_files = filter(lambda x: x[-4:] == ".txt", all_files)
pedestrian_points = []
car_points = []
cyclist_points = []
dontcare_points = []
for file in txt_files:
	f = open("training/labels/" + file, 'r')
	for line in f:
		line = line.split()
		left_x = float(line[4])
		left_y = float(line[5])
		right_x = float(line[6])
		right_y = float(line[7])
		width = abs(right_x - left_x)
		height = abs(right_y - left_y)
		mag = math.sqrt(math.pow(width, 2) + math.pow(height, 2))
		if line[0] == "Pedestrian":
			pedestrian_points.append((width, height))
		elif line[0] == "Car":
			car_points.append((width, height))
		elif line[0] == "Cyclist":
			cyclist_points.append((width, height))
		else:
			dontcare_points.append((width, height))
	f.close()

ped_x, ped_y = zip(*pedestrian_points)
car_x, car_y = zip(*car_points)
cyclist_x, cyclist_y = zip(*cyclist_points)
dontcare_x, dontcare_y = zip(*dontcare_points)
# plt.plot(car_x, car_y, 'bo', ped_x, ped_y, 'ro')

C = 1.0
clf = svm.SVC(kernel='linear', gamma=0.7, C=C)
print ('attempting to fit data')
clf.fit(pedestrian_points+cyclist_points+car_points, [0]*(len(pedestrian_points)+len(cyclist_points))+[1]*len(car_points))
plt.show()

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
    f.close()

img_dir = "testing/images/"
all_files = os.listdir(img_dir)
img_files = filter(lambda x: x[-4:] == ".png", all_files)
result = ""
predict_obj = ""
count = 0
dontcare = 0
success = 0
ped_thresh = 35
for img_fname in img_files:
    print (img_fname)
    result = result + "\n" + img_fname + "\n"
    image = cv2.imread(img_dir+img_fname)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    map_name = img_fname[:-4]
    if (int(map_name) > (3740+200)): # only do 300 images (for now)
        break
    for bb in bb_map[map_name]:
        count += 1
        # print (actual_objs[count])
        if (actual_objs[count] == "DontCare"):
            result = result + actual_objs[count] + " DontCare " + "\n"
            dontcare += 1

        obj = image[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
        obj_width = len(obj)
        obj_height = len(obj[0])
        # resize mask to object size
        mask_car_new = cv2.resize(car_mask, (int(obj_width), int(obj_height)))
        mask_ped_new = cv2.resize(ped_mask, (int(obj_width), int(obj_height)))
        mask_cyc_new = cv2.resize(cyc_mask, (int(obj_width), int(obj_height)))

        obj_sobel = cv2.Sobel(obj, cv2.CV_64F, 0, 1, ksize=5)
        print (obj_width, obj_height)
        res = clf.predict([[obj_width, obj_height]])
        if (res[0] == 0):
            #pedestrian/cyclist
            ped_prod = 0
            for i in range(obj_height):
                for j in range(obj_width):
                    ped_prod += mask_ped_new[i,j,0]*obj[j,i]/255

            if ped_prod > ped_thresh:
                predict_obj = "Pedestrian"
            else:
                predict_obj = "Cyclist"

            result = result + actual_objs[count] + " " + predict_obj + " " + str(ped_prod) + "\n"
            if actual_objs[count] == predict_obj:
                success += 1

        elif(res[0] == 1):
            car_prod = 0; ped_prod = 0; cyc_prod = 0
            for i in range(obj_height):
                for j in range(obj_width):
                    car_prod += mask_car_new[i,j,0]*obj[j,i]/255
                    ped_prod += mask_ped_new[i,j,0]*obj[j,i]/255
                
            # classify object based on car/pedestrian metric
            prods = [car_prod, ped_prod]
            if max(prods) == car_prod:
                predict_obj = "Car"
            elif max(prods) == ped_prod:
                predict_obj = "Pedestrian"
            else: # doesn't do anything right now. to do: set minimum threshold for car/pedestrian
                predict_obj = "Cyclist"
        
            result = result + actual_objs[count] + " " + predict_obj + " " + str(car_prod) + "\n"
            if actual_objs[count] == predict_obj:
                success += 1
        else:
            print ("WTF")

f = open('result.txt', 'w')
f.write(result)
f.close()
count -= dontcare # don't account for "don't cares"
all_success_rate = round(success/count*100, 2)
print ("Total Success: " + str(success) + " / " + str(count) + " = " + str(all_success_rate))
