
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
path = 'S2_result'
data_list = os.listdir(path)
name = '0000'
for data_name in data_list:
    if name in data_name:
        with numpy.load(path + '/' + data_name) as data:
            re_gt = data['re_gt']
            re_in = data['re_in']
            re_out = data['re_out']
            ref = data['ref']


# <codecell> DAIN result viewer
import pickle
from utils import *
with open(r'S2_result/dainflow2_results.pickle','rb') as f:
    re_ledimg_4d = pickle.load(f)
re_ledimg_4d[re_ledimg_4d<0] = 0
re_ledimg_4d[re_ledimg_4d>3] = 3
fig = display_highdimdatacube(re_in[:,:,:,8:],transpose=True)
fig.show()
# with open(r'S2_result/0000_dainflow2_results_ref.pickle','rb') as f:
# ref = pickle.load(f)
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



# <codecell> S3 results viewer
import pickle
from utils import *
import numpy as np
path = 'S3_result'
data_list = os.listdir(path)
name = '0000_spectra__MAX=2'
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
name = '0000_result'

for data_name in data_list:
    if name in data_name:
        with np.load(path + '/' + data_name) as data:
            result = data['result']
            ref = data['ref']
            input = data['input']


fig = display_highdimdatacube(result[:,:,:,0:8],transpose=True)
fig.show()
fig = display_highdimdatacube(ref[:,:,:,0:8],transpose=True)
fig.show()
