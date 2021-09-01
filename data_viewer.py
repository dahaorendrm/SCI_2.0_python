
import matplotlib
matplotlib.use('webagg')
matplotlib.use('Qt5Agg')
%matplotlib qt
%matplotlib inline
# <codecell> S1 results viewer
import pickle
import numpy
import os
from utils import *
path = 'S1_result'
data_list = os.listdir(path)
name = '0000'
for data_name in data_list:
    if name in data_name:
        with numpy.load(path + '/' + data_name) as data:
            gt_outp = data['gt_outp']
            input = data['input']
            output = data['output']
            gt_orig = data['gt_orig']
            if 'gt_leds' in data.keys():
                gt_leds = data['gt_leds']
        break
input
inputimgs = np.squeeze(input)
inputimgs
inputimgs = inputimgs * 255
display_highdimdatacube(inputimgs)

outputimgs = np.squeeze(output)
outputimgs = outputimgs * 255
display_highdimdatacube(outputimgs)

gtimgs = np.squeeze(gt_outp)
gtimgs = gtimgs * 255
display_highdimdatacube(gtimgs)


print(f'Shape of data is inputimgs : {inputimgs.shape}, outputimgs : {outputimgs.shape}, gtimgs : {gtimgs.shape}')
print(f'Max value is inputimgs : {numpy.amax(inputimgs)},  outputimgs : {numpy.amax(outputimgs)},  gtimgs : {numpy.amax(gtimgs)}')
psnr_in = calculate_psnr(gtimgs,inputimgs)
psnr_out = calculate_psnr(gtimgs,outputimgs)
print(f'Input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')





# <codecell> S2 results viewer
import os
import numpy
from utils import *
path = 'S2_result'
data_list = os.listdir(path)
name = '0000_spectra_.npz'
for data_name in data_list:
    if name in data_name:
        with numpy.load(path + '/' + data_name) as data:
            re_gt = data['re_gt']
            re_in = data['re_in']
            re_out = data['re_out']
            ref = data['ref']
fig = display_highdimdatacube(re_out[:,:,:,:8],transpose=True)
fig.show()



# <codecell> S2.5 results viewer
import pickle
from utils import *
import numpy as np
path = 'S2_result'
data_list = os.listdir(path)
name = 'normed0000_spectra'
display = 're_out' #re_gt, re_in, re_out

for data_name in data_list:
    if name in data_name and '.npz' in data_name:
        with np.load(path + '/' + data_name) as data:
            re_display = data[display]
            ref = data['ref']

print(f'The shape of {display} is {re_display.shape}')
fig = display_highdimdatacube(re_display[:,:,:,:8],transpose=True)
fig.show()



# <codecell> S4 results viewer
import pickle
from utils import *
import numpy as np
path = 'S4_result'
data_list = os.listdir(path)
name = '0000_resultpsnr=16.0138'

for data_name in data_list:
    if name in data_name:
        with np.load(path + '/' + data_name) as data:
            result = data['result']
            ref = data['ref']
            input = data['input']


fig = display_highdimdatacube(result[:,:,:,8:],transpose=True)
fig.show()
fig = display_highdimdatacube(ref[:,:,:,8:],transpose=True)
fig.show()
