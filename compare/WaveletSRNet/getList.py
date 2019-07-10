import os

out_file = 'test.list'
dataset = 'dataset/test'

with open(out_file, 'w') as outFile:
    for file in os.listdir(dataset):
        outFile.write(file + ' 0\n')
