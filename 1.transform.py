# -*- coding: utf-8 -*-
import os
import src.Constant as Constant
from src.face_landmark import FaceMarks as Landmark
import cv2
from scipy import misc

DATASET_NAME = "lfw"
destDir = os.path.join(Constant.ROOT_PATH, "datasets",DATASET_NAME + "_trans")
DATASET = os.path.join(Constant.ROOT_PATH, "datasets", DATASET_NAME)
FRONT_DATA = os.path.join(Constant.ROOT_PATH, "datasets", DATASET_NAME + "_front")
landmark = Landmark(faceArea=Constant.FACE_AREA)
OUT_SIZE = 256
REPLACE_OLD = False

if not os.path.isdir(destDir):
    os.mkdir(destDir)
if not os.path.isdir(FRONT_DATA):
    os.mkdir(FRONT_DATA)

totalLength = str(len(os.listdir(DATASET)))
for index, subFolder in enumerate(os.listdir(DATASET)):
    print("processing " + subFolder + " " + str(index+1) + " of total " + totalLength)
    subFolderPath = os.path.join(DATASET, subFolder)
    totalSubFile = str(len(os.listdir(subFolderPath)))
    for i, imgFile in enumerate(os.listdir(subFolderPath)):
        print("processing " + imgFile + " " + str(i+1) + " of total " + totalLength
                    + " in subFolder " + subFolder + " " + str(index+1) + "/" + str(totalLength))
        if not REPLACE_OLD and os.path.exists(os.path.join(destDir, imgFile)):
            continue
        imgPath = os.path.join(subFolderPath, imgFile)
        img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        frontImg = landmark.copyFront(img)
        if frontImg is None:
            transImg = landmark.landmark_transform(img)
            if transImg is None:
                print("transform failed: file %s has no face" % imgFile)
                continue
            resizeImg = misc.imresize(transImg, (OUT_SIZE, OUT_SIZE))
            misc.imsave(os.path.join(destDir, imgFile), resizeImg)
        else:
            resizeImg = misc.imresize(frontImg, (OUT_SIZE, OUT_SIZE))
            misc.imsave(os.path.join(FRONT_DATA, imgFile), resizeImg)
        
