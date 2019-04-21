# -*- coding: utf-8 -*-
import os
import cv2
import dlib

imgPath = os.path.join("datasets", "lfw\\Abdullah_Gul\\Abdullah_Gul_0016.jpg")
SHAPE_MODEL = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
EIGEN_PATH = os.path.join("models", "eigen_face.jpg")


img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

def showPoint(img):
    detector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(SHAPE_MODEL)
    shape = shapePredictor(img, detector(img, 1)[0])
    for i in range(shape.num_parts):
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), 4)
    cv2.imshow(imgPath, img)
    cv2.waitKey()

showPoint(img)