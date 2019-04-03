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

    def landmark_transform(self, img):
        dets = self.detector(img, 1)
        if (len(dets) == 0):
            print("file %s has no face" % file)
            return None
        shape = self.shapePredictor(img, dets[0])

        # Transform position relative to nose center point
        srcPoints = self.getMarkPoints(shape)
        destPoints = self.getRelativePostion(self.getCenter(shape))
        solver = MovingLSQ(srcPoints, destPoints)

        imgIdx = np.zeros((img.shape[0] * img.shape[1], 2))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                imgIdx[i * img.shape[0] + j] = [j, i]
        imgMls = solver.Run_Rigid(imgIdx)
        imgMlsMap = imgMls.reshape((img.shape[0], img.shape[1], 2))
        transImg = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                x = int(math.floor(imgMlsMap[i][j][0]))
                y = int(math.floor(imgMlsMap[i][j][1]))
                if y >= transImg.shape[0] or x >= transImg.shape[1]:
                    continue
                transImg[y, x] = img[i, j]
        return transImg

    def getRelativePostion(self, center):
        destPoints = np.zeros((len(self.eignPoints), 2))
        for index, point in enumerate(self.eignPoints):
            destPoints[index][0] = center[0] + (point[0] - self.eignCenter[0])
            destPoints[index][1] = center[1] + (point[1] - self.eignCenter[1])
        return destPoints
