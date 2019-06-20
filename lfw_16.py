# -*- coding: utf-8 -*- 
import os
import cv2
from scipy import misc
from PIL import Image

sample_path = 'datasets/lfw'
dest_path = sample_path + "/../lfw_16"
middle_path = sample_path + "/../lfw_64"
middleSize = 64
imgSize = 16

if not os.path.exists(dest_path):
    os.mkdir(dest_path)
if not os.path.exists(middle_path):
    os.mkdir(middle_path)

fileList = os.listdir(sample_path)
for index, subFolder in enumerate(fileList):
    print("procesing " + subFolder + " " + str(index+1) + '/' + str(len(fileList)))
    subPath = os.path.join(sample_path, subFolder)
    for file in os.listdir(subPath):
        imgPath = os.path.join(subFolder, file)
        img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        img64 = misc.imresize(img, (middleSize, middleSize), interp='bilinear')
        img16 = misc.imresize(img, (imgSize, imgSize), interp='bilinear')
        misc.imsave(os.path.join(dest_path, file), img16)
        misc.imsave(os.path.join(middle_path, file), img64)