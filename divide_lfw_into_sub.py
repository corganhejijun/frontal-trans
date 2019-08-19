# -*- coding: utf-8 -*- 
import os

lfw_test_path = 'test/'
dest_path = 'lfw_ours_128'

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

fileList = os.listdir(lfw_test_path)
for index, file in enumerate(fileList):
  print("processing " + file + " " + str(index) + " of total " + str(len(fileList)))
  # name: 'Tony_Blair_0114.png'
  subFolderName = file[:-9]
  subFolderPath = os.path.join(dest_path, subFolderName)
  if not os.path.isdir(subFolderPath):
    os.mkdir(subFolderPath)
  os.rename(os.path.join(lfw_test_path, file), os.path.join(subFolderPath, file))
