# -*- coding: utf-8 -*- 
import os
import cv2
from scipy import misc
from PIL import Image
from src.face_landmark import FaceMarks as Landmark
import src.Constant as Constant

sample_path = 'datasets/lfw'
dest_path = sample_path + "/../lfw_16"
middle_path = sample_path + "/../lfw_64"
middleSize = 64
imgSize = 16
landmark = Landmark(faceArea=Constant.HEAD_AREA)

if not os.path.exists(dest_path):
    os.mkdir(dest_path)
if not os.path.exists(middle_path):
    os.mkdir(middle_path)

fileList = os.listdir(sample_path)
for index, subFolder in enumerate(fileList):
    print("procesing " + subFolder + " " + str(index+1) + '/' + str(len(fileList)))
    subPath = os.path.join(sample_path, subFolder)
    for file in os.listdir(subPath):
        imgPath = os.path.join(subPath, file)
        img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        img = landmark.copyFace(img)
        if img is None:
            continue
        img64 = misc.imresize(img, (middleSize, middleSize), interp='bilinear')
        img16 = misc.imresize(img, (imgSize, imgSize), interp='bilinear')
        misc.imsave(os.path.join(dest_path, file), img16)
        misc.imsave(os.path.join(middle_path, file), img64)