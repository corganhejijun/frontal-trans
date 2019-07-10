
import scipy.misc
import os

path = ''
destPath = 'test_gray'
if not os.path.isdir(destPath):
    os.mkdir(destPath)
for file in os.listdir(path):
    img = scipy.misc.imread(os.path.join(path, file), flatten = True).astype(np.uint8)
    scipy.misc.imsave(os.path.join(destPath, file), img)