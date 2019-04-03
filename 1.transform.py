# -*- coding: utf-8 -*-
import os
import src.Constant as Constant
from src.face_landmark import FaceMarks as Landmark
import cv2
from scipy import misc

destDir = os.path.join(Constant.ROOT_PATH, "datasets", "lfw_trans")
DATASET = os.path.join(Constant.ROOT_PATH, "datasets", "lfw")
landmark = Landmark()

if not os.path.isdir(destDir):
    os.mkdir(destDir)

totalLength = str(len(os.listdir(DATASET)))
for index, subFolder in enumerate(os.listdir(DATASET)):
    print("processing " + subFolder + " " + str(index) + " of total " + totalLength)
    subFolderPath = os.path.join(DATASET, subFolder)
    totalSubFile = str(len(os.listdir(subFolderPath)))
    for i, imgFile in enumerate(os.listdir(subFolderPath)):
        print("processing " + imgFile + " " + str(i) + " of total " + totalSubFile + " in subFolder " + subFolder)
        imgPath = os.path.join(subFolderPath, imgFile)
        img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        transImg = landmark.landmark_transform(img)
        outPath = os.path.join(destDir, imgFile)
        misc.imsave(os.path.join(outPath, imgFile), transImg)
