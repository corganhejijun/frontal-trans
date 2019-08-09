# -*- coding: UTF-8 -*-
import os
import cv2
from scipy import misc
from PIL import Image

FILE_PATH = "datalist_gopro.txt"
DATASET_PATH = "../SRN-Deblur/training_set/GOPRO_Large/train"
DEST_SIZE = 128
DEST_PATH = "datasets/gopro"

if not os.path.isdir(DEST_PATH):
    os.mkdir(DEST_PATH)

file = open(FILE_PATH, "r")
line = file.readline()
while len(line) > 0:
    print("processing " + line)
    filenames = line.split()
    fileSharp = os.path.join(DATASET_PATH, filenames[0])
    fileBlur = os.path.join(DATASET_PATH, filenames[1])
    imgSharp = cv2.cvtColor(cv2.imread(fileSharp), cv2.COLOR_BGR2RGB)
    imgBlur = cv2.cvtColor(cv2.imread(fileBlur), cv2.COLOR_BGR2RGB)
    x = 0
    while x < len(imgSharp[0]):
        y = 0
        while y < len(imgSharp):
            imgA = imgSharp[y: y+DEST_SIZE, x : x+DEST_SIZE, :]
            imgB = imgBlur[y: y+DEST_SIZE, x : x+DEST_SIZE, :]
            img = Image.new('RGB', (DEST_SIZE*2, DEST_SIZE))
            img.paste(Image.fromarray(imgA), (0,0))
            img.paste(Image.fromarray(imgB), (DEST_SIZE+1, 0))
            def getFileName(oldName):
                return oldName.replace('/', '_')[:-5] + '_' + str(x) + '_' + str(y) + '.png'
            name = os.path.join(DEST_PATH, getFileName(filenames[0]))
            misc.imsave(name, img)
            y += DEST_SIZE
        x += DEST_SIZE
    line = file.readline()
file.close()