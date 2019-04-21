# -*- coding: utf-8 -*-
import os
import src.Constant as Constant
from src.face_landmark import FaceMarks as Landmark
import cv2
from scipy import misc

destDir = os.path.join(Constant.ROOT_PATH, "datasets", "lfw_trans")
DATASET = os.path.join(Constant.ROOT_PATH, "datasets", "lfw")
landmark = Landmark(faceArea=Constant.FACE_AREA)
OUT_SIZE = 256

if not os.path.isdir(destDir):
    os.mkdir(destDir)

totalLength = str(len(os.listdir(DATASET)))
for index, subFolder in enumerate(os.listdir(DATASET)):
    print("processing " + subFolder + " " + str(index+1) + " of total " + totalLength)
    subFolderPath = os.path.join(DATASET, subFolder)
    totalSubFile = str(len(os.listdir(subFolderPath)))
    for i, imgFile in enumerate(os.listdir(subFolderPath)):
        print("processing " + imgFile + " " + str(i+1) + " of total " + totalSubFile 
                + " in subFolder " + subFolder + " " + str(index+1) + "/" + str(totalLength))
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
            fileName = imgFile[:-len('0001.jpg')] + 'front_' + imgFile[-len('0001.jpg'):]
            misc.imsave(os.path.join(destDir, fileName), resizeImg)
        
