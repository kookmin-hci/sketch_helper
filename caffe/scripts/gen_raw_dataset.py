# Copyright 2017 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import cv2
import numpy as np
import math
import random
import struct
import copy
import sys
from struct import unpack

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break
# main 
total_width, total_height = 255, 255

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-f", "--file", required=True, help="Path to the listfile")
args = vars(ap.parse_args())

# load file list
with open(args["file"]) as f:
        linecount = sum(1 for _ in f)
        f.close()

file_list = open(args["file"], 'r')

#
tf0 = open("temp_Test_stroke.txt", 'w')
tf1 = open("temp_Val_stroke.txt", 'w')
tf2 = open("temp_Train_stroke.txt", 'w')
#

n=1
continue_trig = False
for k in range(linecount):
    file_name = file_list.readline()
    print file_name
    
    mode = 0
    skip_idx = 0
    image_counter = 0 
   
    all_data = []
    train_data_list = []
    test_data_list = []
    val_data_list = []

    for idx, drawing in enumerate(unpack_drawings(file_name[:-1])):
       all_data.append(drawing)

    if len(all_data) <25000:
        print "images smaller than 75000"
        break

    random.shuffle(all_data)
    
    for num_file in range(25000):
        drawing = all_data.pop()
        # reset canvass
        blank_img = np.zeros((total_height,total_width,3), np.uint8)
        blank_img[::] = (255,255,255)
        blank_img2 = np.zeros((total_height,total_width,3), np.uint8)
        blank_img2[::] = (255,255,255)

        result = np.zeros((100,100,3), np.uint8)
        result[::] = (255,255,255)

        xlist = [] 
        ylist = []
        image_list = []

        #min max check for resize & stroke trigger check
        stroke_len = 0
        for i in range(len(drawing['image'])):
            if len(drawing['image'][i][0]) > 1:
                for j in range(len(drawing['image'][i][0])-1):
                    x1 = drawing['image'][i][0][j]
                    x2 = drawing['image'][i][0][j+1]

                    y1 = drawing['image'][i][1][j]
                    y2 = drawing['image'][i][1][j+1]

                    xlist.append(x1)
                    ylist.append(y1)

                    if j==len(drawing['image'][i][0])-2:
                        xlist.append(x2)
                        ylist.append(y2)

                    stroke_len = stroke_len + math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

        min_x = min(xlist)
        min_y = min(ylist)
        max_x = max(xlist)
        max_y = max(ylist)

        #print idx , ":",drawing['key_id']
        stroke_trigger = stroke_len / 5 

        #----------------------------------------        

        #arrange image to center
        diff_y = 255/2 - (max_y - min_y)/2
        diff_x = 255/2 - (max_x - min_x)/2
        #print diff_x,",", diff_y," will moved"
        shift_M = np.float32([[1,0,diff_x], [0,1,diff_y]])

        stroke_len = 0
        # image save
        for i in range(len(drawing['image'])):
            if len(drawing['image'][i][0]) > 1:
                for j in range(len(drawing['image'][i][0])-1):
                    x1 = drawing['image'][i][0][j]
                    x2 = drawing['image'][i][0][j+1]

                    y1 = drawing['image'][i][1][j]
                    y2 = drawing['image'][i][1][j+1]
                    
                    cv2.line(blank_img, (x1,y1), (x2, y2),(0,0,0),2)
                    stroke_len = stroke_len + math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

                    if stroke_len >= stroke_trigger and j <len(drawing['image'][i][0])-2:
                        stroke_len = stroke_len - stroke_trigger 
                        blank_img2[diff_y:max_y+diff_y, diff_x:max_x+diff_x] = cv2.warpAffine(blank_img, shift_M, (255,255))[diff_y:max_y+diff_y, diff_x:diff_x+max_x]

                        result[10:90,10:90] = cv2.resize(blank_img2,(80,80),interpolation = cv2.INTER_CUBIC)
                        image_list.append(copy.copy(result))
            else:
                stroke_len = stroke_len +1
                x1 = drawing['image'][i][0][j]
                y1 = drawing['image'][i][1][j]
                cv.circle(blank_img, (x1,y1), 1,(0,0,0),-1)
        
        shift_M = np.float32([[1,0,diff_x], [0,1,diff_y]])
        blank_img2[diff_y:max_y+diff_y, diff_x:max_x+diff_x] = cv2.warpAffine(blank_img, shift_M, (255,255))[diff_y:max_y+diff_y, diff_x:diff_x+max_x]
        result[10:90,10:90] = cv2.resize(blank_img2,(80,80),interpolation = cv2.INTER_CUBIC)
        image_list.append(copy.copy(result))
        if len(image_list) < 5:
            for i in range(5-len(image_list)):
                image_list.append(copy.copy(result))
        if len(image_list) != 5:
            print "img length is wrong: ", len(image_list)
            continue

        #####image list update
        for r in range(len(image_list)):
            cv2.imwrite('./imgData/'+str(k)+'/'+file_name[10:len(file_name)-5]+'_'+ str(image_counter) +'_'+str(r)+".png",image_list[r])

        if num_file < 2500: 
            tf0.write("imgData/"+str(k)+'/'+file_name[10:len(file_name)-5]+'_'+str(image_counter)+"*" + str(k)+'\n')

        elif num_file < 5000:
            tf1.write("imgData/"+str(k)+'/'+file_name[10:len(file_name)-5]+'_'+str(image_counter)+"*" + str(k)+'\n')

        else:
            tf2.write("imgData/"+str(k)+'/'+file_name[10:len(file_name)-5]+'_'+str(image_counter)+"*" + str(k)+'\n')

    
        print file_name + ": "+str(num_file)
        image_counter = image_counter +1

tf0.close()
tf1.close()
tf2.close()
file_list.close()
#
######image list shuffle between class
#
tf0 = open("temp_Test_stroke.txt", 'r')
tf1 = open("temp_Val_stroke.txt", 'r')
tf2 = open("temp_Train_stroke.txt", 'r')

f0 = open("75000_Test_stroke.txt", 'w')
f1 = open("75000_Val_stroke.txt", 'w')
f2 = open("75000_Train_stroke.txt", 'w')

test_data_list = []
train_data_list = []
val_data_list = []

# test data naming
for i in range(2500*345):
    test_data_list.append(tf0.readline())    

random.shuffle(test_data_list)
for i in range(2500*345):
    line = test_data_list.pop()    
    sp_line = line.split('*')

    for r in range(5):
        f0.write(sp_line[0]+'_'+str(r)+".png "+sp_line[1])

# val data naming
for i in range(2500*345):
    val_data_list.append(tf1.readline())    

random.shuffle(val_data_list)
for i in range(2500*345):
    line = val_data_list.pop()    
    sp_line = line.split('*')

    for r in range(5):
        f1.write(sp_line[0]+'_'+str(r)+".png "+sp_line[1])

# train_data naming
for i in range(20000*345):
    train_data_list.append(tf2.readline())    

random.shuffle(train_data_list)
for i in range(20000*345):
    line = train_data_list.pop()    
    sp_line = line.split('*')

    for r in range(5):
        f2.write(sp_line[0]+'_'+str(r)+".png "+sp_line[1])

f0.close()
f1.close()
f2.close()
tf0.close()
tf1.close()
tf2.close()
