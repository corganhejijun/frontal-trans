# -*- coding: UTF-8 -*-
import os
from scipy import misc
from PIL import Image

test_path = 'test'
target_path = 'gopro_combine'

if not os.path.isdir(target_path):
  os.mkdir(target_path)

fileList = os.listdir(test_path)
fileList.sort()

for index, file in fileList:
  print('processing ' + file + ' ' + str(index) +  ' of total ' + str(len(fileList)))
  # file name format: 'GOPR0410_11_00_sharp_000118_768_128.png'
  file_pre_name = file[:27]
  ext = file[-4:]
  sizeStrs = file[28:-4].split('_')
  x = int(sizeStrs[0])
  y = int(sizeStrs[1])
  target_name = os.path.join(target_path, file_pre_name + ext)
  fileImg = Image.open(os.path.join(test_path, target_name))
  img = None
  if os.path.isfile(target_name):
    img = Image.open(target_name)
  else:
    img = Image.new('RGB', (1280,768))
  img.paste(fileImg, (x, y))
  misc.imsave(target_name, img)
