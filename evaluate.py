import numpy as np
import os
import cv2
import skimage

training_data_path = os.path.join('datasets', 'celeb_train', 'val_test')
test_data_path = 'test'
out_file = 'out.csv'

with open(out_file, 'w') as outFile:
    outFile.write('fileName, linear Psnr, bicubic Psnr, GAN Psnr, linear Ssim, bicubic Ssim, GAN Ssim\n')
    for file in os.listdir(test_data_path):
        if not file.endswith('.png'):
            continue
        print("processing " + file)
        testFile = os.path.join(test_data_path, file)
        combineFile = os.path.join(training_data_path, file[:-4] + '.jpg')
        testImg = cv2.cvtColor(cv2.imread(testFile), cv2.COLOR_BGR2RGB)
        combineImg = cv2.cvtColor(cv2.imread(combineFile), cv2.COLOR_BGR2RGB)
        imgWidth = testImg.shape[1]
        originalImg = combineImg[:, :imgWidth]
        gtImg = combineImg[:, imgWidth:]
        smallImg = cv2.resize(gtImg, (int(imgWidth/4), int(imgWidth/4)), interpolation=cv2.INTER_CUBIC)
        bicubicImg = cv2.resize(smallImg, (imgWidth, imgWidth), interpolation=cv2.INTER_CUBIC)

        originPsnr = skimage.measure.compare_psnr(originalImg, gtImg)
        testPsnr = skimage.measure.compare_psnr(testImg, gtImg)
        bicubicPsnr = skimage.measure.compare_psnr(bicubicImg, gtImg)
        originSsim = skimage.measure.compare_ssim(originalImg, gtImg, multichannel=True)
        testSsim = skimage.measure.compare_ssim(testImg, gtImg, multichannel=True)
        bicubicSsim = skimage.measure.compare_ssim(bicubicImg, gtImg, multichannel=True)
        outFile.write(file[:-4] + ', ' 
                        + str(originPsnr) + ', ' + str(bicubicPsnr) + ', ' + str(testPsnr) + ', '
                        + str(originSsim) + ', ' + str(bicubicSsim) + ', ' + str(testSsim) 
                        + '\n')
    