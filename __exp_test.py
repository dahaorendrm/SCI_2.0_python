import scipy.io as scio
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform as skitrans
from pathlib import Path
import tifffile

from S0_gaptv import run_gap_tv as S0run


# S0
pool = multiprocessing.Pool()
mask = scio.loadmat(PATH/'mask.mat')['mask']
PATH = Path('./expdata')
MODEL = 'lesti_sst'
numf = mask.shape[2]
dataset = []
datalist = os.listdir(PATH)
for idx,name in enumerate(datalist):
    if idx > 10: # small test sets
        break
    mea = scio.loadmat(PATH/name)['img']
    dataset.append((mea,mask,numf))
#S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
return_crops_data = pool.starmap(S0run.compressive_model_exp, comp_input)
if not os.path.exists('S0_gaptv/data/exp'):
    os.mkdir('S0_gaptv/data/exp')
    os.mkdir('S0_gaptv/data/exp/mea')
    os.mkdir('S0_gaptv/data/exp/img_n')
for idx,(mea,re) in enumerate(return_crops_data):
    tifffile.imwrite(mea,Path('S0_gaptv/data/exp/mea')/(datalist[idx]+'.tiff'))
    tifffile.imwrite(re,Path('S0_gaptv/data/exp/img_n')/(datalist[idx]+'.tiff'))

# S1
