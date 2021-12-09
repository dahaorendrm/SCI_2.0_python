import scipy.io as scio
import os
import numpy as np
import tifffile,pickle
import multiprocessing,threading,queue
import PIL
import itertools as itert
import time
from collections import namedtuple
import datetime
from pathlib import Path


from S0_gaptv import run_gap_tv
from S1_denoiser import test as S1test
from S2_flow_predict import test as S2test
from S3_spectra_convert import test as S3test

pool = multiprocessing.Pool()
MODEL = 'lesti_sst'
COMP_FRAME = 24 # 16,24,32
imgs = scio.loadmat('4D_Lego.mat')['img']
print(f'Input LEGO data max is {np.amax(imgs)}.')
#print(f'shape of imgs is {imgs.shape}')
crops = []
for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
    #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
    crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
comp_input = [(MODEL,crop) for crop in crops]
return_crops_data = pool.starmap(run_gap_tv.compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
crops_mea = []
crops_img = []
crops_led = []
for (orig_leds,mea,re) in return_crops_data:
    crops_mea.append(mea)
    crops_img.append(re)
    crops_led.append(orig_leds)
### (orig_leds,mea,re)
run_gap_tv.save_crops('./S0_gaptv/data/test',0,'4D_lego',crops_mea,crops_img, crops_gt=crops, crops_led=crops_led)

#S1test.test('./S0_gaptv/data/test/')
#S2test.test('./S1_denoiser/result')
S2test.test('./S0_gaptv/data/test/img_n')
S3test.test('./S2_flow_predict/result/re','./S0_gaptv/data/test/gt')
