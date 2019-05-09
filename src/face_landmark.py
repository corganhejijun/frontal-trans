# -*- coding: utf-8 -*-
from .Constant import *
import os
import numpy as np
import math
import cv2
import dlib
from .MovingLSQ import MovingLSQ

class FaceMarks:
    def __init__(self, faceArea=FACE_AREA):
        self.SHAPE_MODEL = os.path.join(ROOT_PATH, "models", "shape_predictor_68_face_landmarks.dat")
        self.EIGEN_PATH = os.path.join(ROOT_PATH, "models", "eigen_face.jpg")
        self.eignImg = cv2.cvtColor(cv2.imread(self.EIGEN_PATH), cv2.COLOR_BGR2RGB)
        self.detector = dlib.get_frontal_face_detector()
        self.shapePredictor = dlib.shape_predictor(self.SHAPE_MODEL)
        self.eignShape = self.shapePredictor(self.eignImg, self.detector(self.eignImg, 1)[0])
        self.eignPoints = self.getMarkPoints(self.eignShape)
        self.eignCenter = self.getCenter(self.eignShape)
        self.faceArea = faceArea

    def getMarkPoints(self, shape):
        points = np.zeros((shape.num_parts, 2))
        for i in range(shape.num_parts):
            points[i][0] = shape.part(i).x
            points[i][1] = shape.part(i).y
        return points

    def getCenter(self, shape):
        NOSE_CENTER = 28
        return [shape.part(NOSE_CENTER).x, shape.part(NOSE_CENTER).y]

    def getFaceArea(self, img, shape, srcPoints, dstPoints):
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
        if self.faceArea == HEAD_AREA:
            # add forehead
            yMin -= math.floor((yMax-yMin)/2)
            # add ears
            xMin -= math.floor((xMax-xMin)/6)
            xMax += math.floor((xMax-xMin)/6)
        def getPointRange(points, xMin, xMax, yMin, yMax):
            for i in range(points.shape[0]):
                if xMin > points[i][0]:
                    xMin = points[i][0]
                if xMax < points[i][0]:
                    xMax = points[i][0]
                if yMin > points[i][1]:
                    yMin = points[i][1]
                if yMax < points[i][1]:
                    yMax = points[i][1]
            return xMin, xMax, yMin, yMax
        xMin, xMax, yMin, yMax = getPointRange(srcPoints, xMin, xMax, yMin, yMax)
        xMin, xMax, yMin, yMax = getPointRange(dstPoints, xMin, xMax, yMin, yMax)
        if yMin < 1:
            yMin = 1
        if xMin < 1:
            xMin = 1
        if yMax > len(img) - 1:
            yMax = len(img) - 1
        if xMax > len(img[0]) - 1:
            xMax = len(img[0]) - 1
        return img[int(yMin)-1:int(yMax)+1, int(xMin)-1:int(xMax)+1, :], int(xMin)-1, int(yMin)-1

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

    def getMargin(self, transMap):
        left = 0
        right = transMap.shape[1]
        top = 0
        bottom = transMap.shape[0]
        for i in range(transMap.shape[0]):
            for j in range(transMap.shape[1]):
                x = int(math.floor(transMap[i][j][0]))
                y = int(math.floor(transMap[i][j][1]))
                if left > x:
                    left = x
                if right < x:
                    right = x
                if top > y:
                    top = y
                if bottom < y:
                    bottom = y
                x = int(math.ceil(transMap[i][j][0]))
                y = int(math.ceil(transMap[i][j][1]))
                if left > x:
                    left = x
                if right < x:
                    right = x
                if top > y:
                    top = y
                if bottom < y:
                    bottom = y
        return 0-left, right-transMap.shape[1], 0-top, bottom-transMap.shape[0]
    
    def fillTransImg(self, img, transMap):
        leftMargin, rightMargin, topMargin, bottomMargin = self.getMargin(transMap)
        imgWidth = img.shape[1] + leftMargin + rightMargin
        imgHeight = img.shape[0] + topMargin + bottomMargin
        transImg = np.zeros((imgHeight, imgWidth, 3))
        transFlag = np.zeros((imgHeight, imgWidth))
        for i in range(transMap.shape[0]):
            for j in range(transMap.shape[1]):
                x = int(math.floor(transMap[i][j][0])) + leftMargin
                y = int(math.floor(transMap[i][j][1])) + topMargin
                if y >= imgHeight or x >= imgWidth:
                    continue
                if transFlag[y, x] == 0:
                    transImg[y, x] = img[i, j]
                    transFlag[y, x] = 1
                else:
                    # get the first pixel value or average value
                    continue
                    # transImg[y, x] = (transImg[y, x] + img[i, j])/2
                x = int(math.ceil(transMap[i][j][0])) + leftMargin
                y = int(math.ceil(transMap[i][j][1])) + topMargin
                if y >= imgHeight or x >= imgWidth:
                    continue
                if transFlag[y, x] == 0:
                    transImg[y, x] = img[i, j]
                    transFlag[y, x] = 1
                else:
                    # get the first pixel value or average value
                    continue
                    # transImg[y, x] = (transImg[y, x] + img[i, j])/2
        # Fill the holes
        for i in range(imgHeight):
            for j in range(imgWidth):
                if transFlag[i, j] > 0:
                    continue
                x, y= self.getNearbyPixel(transFlag, j, i)
                if x is None:
                    transImg[i, j] = img[i, j]
                else:
                    transImg[i, j] = transImg[y, x]
        return transImg

    def square(self, img):
        width = img.shape[1]
        height = img.shape[0]
        if width > height:
            squareImg = np.zeros((width, width, 3))
            top = math.ceil((width - height)/2)
            bottom = math.floor((width - height)/2)
            if bottom == 0:
                squareImg[top:squareImg.shape[0],:,:] = img
            else:
                squareImg[top:-bottom,:,:] = img
            for i in range(top):
                squareImg[i,:,:] = img[0,:,:]
            for i in range(bottom):
                squareImg[-i-1,:,:] = img[-1,:,:]
            return squareImg
        elif height > width:
            squareImg = np.zeros((height, height, 3))
            left = math.ceil((height-width)/2)
            right = math.floor((height-width)/2)
            if right == 0:
                squareImg[:, left:squareImg.shape[1], :] = img
            else: 
                squareImg[:, left:-right, :] = img
            for i in range(left):
                squareImg[:,i,:] = img[:,0,:]
            for i in range(right):
                squareImg[:,-i-1,:] = img[:,-1,:]
            return squareImg
        return img

    def getFaceImgPosition(self, srcPoints, dstPoints, left, top):
        toDel = []
        for i in range(srcPoints.shape[0]):
            srcPoints[i][0] -= left
            srcPoints[i][1] -= top
            dstPoints[i][0] -= left
            dstPoints[i][1] -= top
            if srcPoints[i][0] < 0 or srcPoints[i][1] < 0 or dstPoints[i][0] < 0 or dstPoints[i][1] < 0:
                toDel.append(i)
        srcPoints = np.delete(srcPoints, toDel, 0)
        dstPoints = np.delete(dstPoints, toDel, 0)
        return srcPoints, dstPoints

    def landmark_transform(self, img):
        dets = self.detector(img, 1)
        if (len(dets) == 0):
            return None
        shape = self.shapePredictor(img, dets[0])

        # Transform position relative to nose center point
        srcPoints = self.getMarkPoints(shape)
        destPoints = self.getRelativePostion(self.getCenter(shape))

        faceImg, left, top = self.getFaceArea(img, shape, srcPoints, destPoints)
        srcPoints, destPoints = self.getFaceImgPosition(srcPoints, destPoints, left, top)
        # solver = Transform(srcPoints, destPoints, (0, 0, faceImg.shape[1], faceImg.shape[0]))
        solver = MovingLSQ(srcPoints, destPoints)

        # imgIdx = np.zeros((faceImg.shape[0], faceImg.shape[1], 2))
        imgIdx = np.zeros((faceImg.shape[0] * faceImg.shape[1], 2))
        for i in range(faceImg.shape[0]):
            for j in range(faceImg.shape[1]):
                # imgIdx[i, j] = [j, i]
                imgIdx[i * faceImg.shape[1] + j] = [j, i]
        # transMap = solver.Run(imgIdx, faceImg)
        transMap = solver.Run_Rigid(imgIdx)
        transMap = transMap.reshape((faceImg.shape[0], faceImg.shape[1], 2))

        transImg = self.fillTransImg(faceImg, transMap)
        return self.square(transImg)
    
    def copyFront(self, img):
        dets = self.detector(img, 1)
        if (len(dets) == 0):
            return None
        shape = self.shapePredictor(img, dets[0])
        points = self.getMarkPoints(shape)
        totalDistance = 0
        for i in range(points.shape[0]):
            distance = (abs(points[i][0] - self.eignPoints[i][0])
                        + abs(points[i][1] - self.eignPoints[i][1]))
            totalDistance += distance
        if totalDistance > 100:
            return None
        faceImg = self.getFaceArea(img, shape, points, points)
        return faceImg


    def getRelativePostion(self, center):
        destPoints = np.zeros((len(self.eignPoints), 2))
        for index, point in enumerate(self.eignPoints):
            destPoints[index][0] = center[0] + (point[0] - self.eignCenter[0])
            destPoints[index][1] = center[1] + (point[1] - self.eignCenter[1])
        return destPoints
