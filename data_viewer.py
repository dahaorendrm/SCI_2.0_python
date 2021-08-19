import pickle
import numpy
import os
import matplotlib
matplotlib.use('qt5agg')

# chasti result viewer
path = 'test/result'
data_list = os.listdir(path)
name = '0003'
for data_name in data_list:
    if name in data_name:
        with numpy.load(path + '/' + data_name) as data:
            gt_outp = data['gt_outp']
            input = data['input']
            output = data['output']
            gt_orig = data['gt_orig']
            if 'gt_leds' in data.keys():
                gt_leds = data['gt_leds']
    # if name in data and 'input' in data:
    #     with open(path + '/' + data, 'rb') as f:
    #         inputimgs = numpy.load(f)
    #
    # if name in data and 'result' in data:
    #     with open(path + '/' + data, 'rb') as f:
    #         outputimgs = numpy.load(f)
    #
    # if name in data and 'gt' in data:
    #     with open(path + '/' + data, 'rb') as f:
    #         gtimgs = numpy.load(f)
from utils import *

inputimgs = np.squeeze(input)
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


# <codecell> DAIN result viewer
import pickle
from utils import *
with open(r'S2_result/dainflow2_results.pickle','rb') as f:
    re_ledimg_4d = pickle.load(f)
with open(r'S2_result/0000_dainflow2_results_ref.pickle','rb') as f:
    ref = pickle.load(f)
re_ledimg_4d[re_ledimg_4d<0] = 0
re_ledimg_4d[re_ledimg_4d>5] = 5
fig = display_highdimdatacube(re_ledimg_4d[:,:,:,:8],transpose=True)
fig.show()
# fig_ref = display_highdimdatacube(ref,transpose=True)
# fig_ref.show()


from scipy import signal
import numpy as np
led_curve = signal.resample(mea.led_curve,8,axis=0)
orig = signal.resample(mea.orig,8,axis=2)
temp = np.moveaxis(re_ledimg_4d,-1,-2)
shape_ = temp.shape
temp = np.reshape(temp,(np.cumprod(shape_[:3])[2],shape_[3]))
temp = np.linalg.solve(led_curve.transpose(), temp.transpose())
temp = np.reshape(temp.transpose(),shape_)
temp = np.moveaxis(temp,-1,-2)
fig = display_highdimdatacube(temp[:,:,:,:8],transpose=True)
fig.show()
fig_ref = display_highdimdatacube(orig[:,:,:,:8],transpose=True)
fig_ref.show()
