# -*- coding: utf-8 -*- 
import os
import cv2
from scipy import misc
from PIL import Image

sample_path = '../git/Face-and-Image-super-resolution/testset'
dest_path = 'datasets/SRtrain/val_test/'
imgSize = 64

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

fileList = os.listdir(sample_path)
for index, file in enumerate(fileList):
    imgPath = os.path.join(sample_path, file)
    if os.path.isdir(imgPath):
        continue
    print("procesing " + file + " " + str(index+1) + '/' + str(len(fileList)))
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img = misc.imresize(img, (imgSize, imgSize), interp='bilinear')
    combineImg = Image.new('RGB', (img.shape[0]*2, img.shape[0]))
    combineImg.paste(Image.fromarray(img), (0,0))
    combineImg.paste(Image.fromarray(img), (img.shape[0]+1,0))
    misc.imsave(os.path.join(dest_path, file))
