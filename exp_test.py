import scipy.io as scio
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform as skitrans
from pathlib import Path
import tifffile
import multiprocessing

from S0_gaptv import run_gap_tv as S0run
from S1_denoiser import test as S1run
from S2_flow_predict import test as S2run
from S3_spectra_convert import test as S3run
# S0
def S0run_test():
    pool = multiprocessing.Pool()
    PATH = Path('./expdata')
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    MODEL = 'lesti_sst'
    numf = mask.shape[2]
    dataset = []
    datalist = os.listdir(PATH)
    for idx,name in enumerate(datalist):
        if idx > 20: # small test sets
            break
        mea = scio.loadmat(PATH/name)['img']
        dataset.append((MODEL,mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = pool.starmap(S0run.compressive_model_exp, dataset)
    if not os.path.exists('S0_gaptv/data/exp'):
        os.mkdir('S0_gaptv/data/exp')
        os.mkdir('S0_gaptv/data/exp/mea')
        os.mkdir('S0_gaptv/data/exp/img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(Path('S0_gaptv/data/exp/mea')/(datalist[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(Path('S0_gaptv/data/exp/img_n')/(datalist[idx][:-4]+'.tiff'),re)

if __name__=='__main__':
    # S0
    #S0run_test()
    # S1
    # S1run.test('S0_gaptv/data/exp','S1_denoiser/result/exp','expdata/mask.mat')
    # S2
    #S2run.test('S0_gaptv/data/exp/img_n','S0_gaptv/data/exp/','S2_flow_predict/result/exp/')
    # S3
    S3run.test('S2_flow_predict/result/exp/re','S0_gaptv/data/exp/', 'S3_spectra_convert/result')
