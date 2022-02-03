import pickle
import numpy
import os

path = 'test/result'
data_list = os.listdir(path)
name = '0000'
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
