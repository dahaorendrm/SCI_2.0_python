
import matplotlib
matplotlib.use('webagg')
matplotlib.use('Qt5Agg')
%matplotlib qt
%matplotlib inline

# <codecell> S0 gaptv result viewer
import tifffile
import matplotlib
import numpy as np
from utils import *
name = '4D_lego_0000_.tiff'
path = './data/data/test/feature/' + name
re_gaptv = tifffile.imread(path)
re_gaptv = re_gaptv[...,1:]
path = './data/data/test/gt/' + name
gt = tifffile.imread(path)
gt = gt[...,1:]
path = './data/data/test/gt_led/' + name
orig = tifffile.imread(path)
ind_c = 0
orig_ = []
CHAN = 8
for ind in range(orig.shape[-1]):
    temp = orig[...,ind_c,ind]
    orig_.append(temp)
    ind_c = ind_c+1 if ind_c<CHAN-1 else 0
orig = np.stack(orig_,1)
orig = np.moveaxis(orig,1,-1)

MAX_v = np.amax(orig)
print(f'Maxv of gt is {np.amax(gt)}, of orig is {MAX_v}, of recon is {np.amax(re_gaptv)}')
psnr_v = calculate_psnr(orig,re_gaptv)
print(f'psnr before: {psnr_v}')
re_gaptv_max = re_gaptv/np.amax(orig)
orig_max = orig/np.amax(orig)
psnr_v = calculate_psnr(orig_max,re_gaptv_max)
print(f'psnr after divid max: {psnr_v}')
re_gaptv1 = re_gaptv
re_gaptv1[re_gaptv1>MAX_v] = MAX_v
psnr_v = calculate_psnr(orig,re_gaptv1,MAX_v)
print(f'psnr after take off max: {psnr_v}')
re_gaptv_max = re_gaptv/np.amax(re_gaptv)
orig_max = orig/np.amax(orig)
psnr_v = calculate_psnr(orig_max,re_gaptv_max,1)
print(f'psnr after divid own max: {psnr_v}')

display_highdimdatacube(re_gaptv)


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
