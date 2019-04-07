# -*- coding: utf-8 -*-
from .Constant import ROOT_PATH
import os
from .MovingLSQ import MovingLSQ
import numpy as np
import math
import cv2
import dlib

class FaceMarks:
    def __init__(self):
        self.SHAPE_MODEL = os.path.join(ROOT_PATH, "models", "shape_predictor_68_face_landmarks.dat")
        self.EIGEN_PATH = os.path.join(ROOT_PATH, "datasets", "eigen_face.jpg")
        self.eignImg = cv2.cvtColor(cv2.imread(self.EIGEN_PATH), cv2.COLOR_BGR2RGB)
        self.detector = dlib.get_frontal_face_detector()
        self.shapePredictor = dlib.shape_predictor(self.SHAPE_MODEL)
        self.eignShape = self.shapePredictor(self.eignImg, self.detector(self.eignImg, 1)[0])
        self.eignPoints = self.getMarkPoints(self.eignShape)
        self.eignCenter = self.getCenter(self.eignShape)

    def getMarkPoints(self, shape):
        LANDMARK_LIST = [0, 2, 4,   # right face
                            8,      # jaw
                         12, 14, 16,# left face
                         36, 39,    # right eye
                         42, 45,    # left eye
                         31, 35,    # nose
                         49, 55]    # mouth
        points = np.zeros((len(LANDMARK_LIST), 2))
        for index, i in enumerate(LANDMARK_LIST):
            points[index][0] = shape.part(i).x
            points[index][1] = shape.part(i).y
        return points

    def getCenter(self, shape):
        NOSE_CENTER = 28
        return [shape.part(NOSE_CENTER).x, shape.part(NOSE_CENTER).y]

    def getFaceArea(self, img, shape):
        xMin = len(img[0])
        xMax = 0
        yMin = len(img)
        yMax = 0
        for i in range(shape.num_parts):
            if (shape.part(i).x < xMin):
                xMin = shape.part(i).x
            if (shape.part(i).x > xMax):
                xMax = shape.part(i).x
            if (shape.part(i).y < yMin):
                yMin = shape.part(i).y
            if (shape.part(i).y > yMax):
                yMax = shape.part(i).y
        return img[yMin:yMax, xMin:xMax, :]

    def getNearbyPixel(self, flag, x, y):
        index = 0
        while True:
            index += 1
            if index + y >= flag.shape[0] and y - index < 0 and index + x >= flag.shape[1] and x - index < 0:
                break
            for i in range(index):
                if y + i < flag.shape[0] and x + index < flag.shape[1] and flag[y + i, x + index] > 0:
                    return x + index, y + i 
                if y - i >= 0 and x + index < flag.shape[1] and flag[y - i, x + index] > 0:
                    return x + index, y - i
                if y + i < flag.shape[0] and x - index >= 0 and flag[y + i, x - index] > 0:
                    return x - index, y + i
                if y - i >= 0 and x - index >= 0 and flag[y - i, x - index] > 0:
                    return x - index, y - i
                if y + index < flag.shape[0] and x + i < flag.shape[1] and flag[y + index, x + i] > 0:
                    return x + i, y + index
                if y + index < flag.shape[0] and x - i >= 0 and flag[y + index, x - i] > 0:
                    return x - i, y + index
                if y - index >= 0 and x + i < flag.shape[1] and flag[y - index, x + i] > 0:
                    return x + i, y - index
                if y - index >= 0 and x - i >= 0 and flag[y - index, x - i] > 0:
                    return x - i, y - index
        return None, None
    
    def fillTransImg(self, img, mlsMap):
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        transImg = np.zeros(img.shape)
        transFlag = np.zeros((imgHeight, imgWidth))
        for i in range(imgHeight):
            for j in range(imgWidth):
                x = int(math.floor(mlsMap[i][j][0]))
                y = int(math.floor(mlsMap[i][j][1]))
                if y >= imgHeight or x >= imgWidth:
                    continue
                if transFlag[y, x] == 0:
                    transImg[y, x] = img[i, j]
                    transFlag[y, x] = 1;
                else:
                    transImg[y, x] = (transImg[y, x] + img[i, j])/2
                x = int(math.ceil(mlsMap[i][j][0]))
                y = int(math.ceil(mlsMap[i][j][1]))
                if y >= imgHeight or x >= imgWidth:
                    continue
                if transFlag[y, x] == 0:
                    transImg[y, x] = img[i, j]
                    transFlag[y, x] = 1;
                else:
                    transImg[y, x] = (transImg[y, x] + img[i, j])/2
        # Fill the holes
        for i in range(imgHeight):
            for j in range(imgWidth):
                if transFlag[i, j] > 0:
                    continue
                x, y= self.getNearbyPixel(transFlag, j, i)
                if x == None:
                    transImg[i, j] = img[i, j]
                else:
                    transImg[i, j] = transImg[y, x]
        return transImg

    def landmark_transform(self, img):
        dets = self.detector(img, 1)
        if (len(dets) == 0):
            print("file %s has no face" % file)
            return None
        shape = self.shapePredictor(img, dets[0])
        faceImg = self.getFaceArea(img, shape)

        # Transform position relative to nose center point
        srcPoints = self.getMarkPoints(shape)
        destPoints = self.getRelativePostion(self.getCenter(shape))
        solver = MovingLSQ(srcPoints, destPoints)

        imgIdx = np.zeros((faceImg.shape[0] * faceImg.shape[1], 2))
        for i in range(faceImg.shape[0]):
            for j in range(faceImg.shape[1]):
                imgIdx[i * faceImg.shape[1] + j] = [j, i]
        imgMls = solver.Run_Rigid(imgIdx)
        imgMlsMap = imgMls.reshape((faceImg.shape[0], faceImg.shape[1], 2))
        return self.fillTransImg(faceImg, imgMlsMap)

    def getRelativePostion(self, center):
        destPoints = np.zeros((len(self.eignPoints), 2))
        for index, point in enumerate(self.eignPoints):
            destPoints[index][0] = center[0] + (point[0] - self.eignCenter[0])
            destPoints[index][1] = center[1] + (point[1] - self.eignCenter[1])
        return destPoints
