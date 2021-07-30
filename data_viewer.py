import pickle
import numpy
import os

path = 'test/result'
data_list = os.listdir(path)
name = '0004'
for data in data_list:
    if name in data and 'input' in data:
        with open(path + '/' + data, 'rb') as f:
            inputimgs = numpy.load(f)

    if name in data and 'result' in data:
        with open(path + '/' + data, 'rb') as f:
            outputimgs = numpy.load(f)

    if name in data and 'gt' in data:
        with open(path + '/' + data, 'rb') as f:
            gtimgs = numpy.load(f)


from utils import *

inputimgs = np.squeeze(inputimgs)
display_highdimdatacube(inputimgs)

outputimgs = np.squeeze(outputimgs)
display_highdimdatacube(outputimgs)

gtimgs = np.squeeze(gtimgs)
display_highdimdatacube(gtimgs)


print(f'Shape of data is inputimgs : {inputimgs.shape}, outputimgs : {outputimgs.shape}, gtimgs : {gtimgs.shape}')
print(f'Max value is inputimgs : {numpy.amax(inputimgs)},  outputimgs : {numpy.amax(outputimgs)},  gtimgs : {numpy.amax(gtimgs)}')
psnr_in = calculate_psnr(gtimgs,inputimgs)
psnr_out = calculate_psnr(gtimgs,outputimgs)
print(f'Input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')
