# -*- coding: UTF-8 -*-
import os
from PIL import Image
from scipy import misc

combine_path = 'gopro_combine'
result_path = 'gopro_crop'

if not os.path.isdir(result_path):
  os.mkdir(result_path)

for file in os.listdir(combine_path):
  print('processing ' + file)
  img = Image.open(os.path.join(combine_path, file))
  misc.imsave(os.path.join(result_path, file),
              img.crop((0,0,1280,720)))