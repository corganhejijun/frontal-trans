# -*- coding: utf-8 -*- 
import os
import cv2
from scipy import misc
from PIL import Image

sample_path = 'datasets/lfw'
dest_path = 'datasets/lfw_test'
middleSize = 32
imgSize = 128
kernel_size = (5, 5)
sigma = 5

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

lfw_list = os.listdir(sample_path)
for index, subFolder in enumerate(lfw_list):
    print("procesing " + subFolder + " " + str(index+1) + '/' + str(len(lfw_list)))
    fileList = os.listdir(os.path.join(sample_path, subFolder))
    for file in fileList:
        imgPath = os.path.join(sample_path, subFolder, file)
        if os.path.isdir(imgPath):
            continue
        imgB = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        imgA = misc.imresize(imgB, (middleSize, middleSize), interp='bilinear')
        imgA = misc.imresize(imgA, (imgSize, imgSize), interp='bilinear')
        combineImg = Image.new('RGB', (img.shape[0]*2, img.shape[0]))
        combineImg.paste(Image.fromarray(imgA), (0,0))
        combineImg.paste(Image.fromarray(imgB), (img.shape[0]+1,0))
        misc.imsave(os.path.join(dest_path, file), combineImg)
