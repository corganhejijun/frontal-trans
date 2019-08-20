# -*- coding: utf-8 -*- 
import os

lfw_data_path = 'compare/lfw_128/test'
dest_path = 'compare/lfw_128/test_'
gt_path = 'compare/lfw_128/gt'

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

fileList = os.listdir(lfw_data_path)
for index, file in enumerate(fileList):
  print("processing " + file + " " + str(index) + " of total " + str(len(fileList)))
  # name: 'Tony_Blair_0114.png'
  subFolderName = file[:-4]
  subFolderPath = os.path.join(dest_path, subFolderName)
  os.mkdir(subFolderPath)
  os.rename(os.path.join(lfw_test_path, file), os.path.join(subFolderPath, file))
  os.rename(os.path.join(gt_path, file), os.path.join(subFolderName, file[:-4] + '_gt' + file[-4:]))
