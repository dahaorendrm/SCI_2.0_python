import scipy.io as scio
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform as skitrans
from pathlib import Path
import tifffile
import multiprocessing

import S0_run as S0run
import S1_test as S1run
import S2_test as S2run
import S3_test as S3run
# S0
def S0run_test():
    pool = multiprocessing.Pool()
    PATH = Path('./expdata')
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    MODEL = 'lesti_sst'
    numf = mask.shape[2]
    dataset = []
    datalist = os.listdir(PATH)
    dataout = []
    for idx,name in enumerate(datalist):
        if not 'Lego0019' in name: # small test sets
            print(name)
            continue
        mea = scio.loadmat(PATH/name)['img']
        dataout.append(name)
        dataset.append((MODEL,mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = pool.starmap(S0run.compressive_model_exp, dataset)
    if not os.path.exists('S0_gaptv/data/exp'):
        os.mkdir('S0_gaptv/data/exp')
        os.mkdir('S0_gaptv/data/exp/mea')
        os.mkdir('S0_gaptv/data/exp/img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(Path('S0_gaptv/data/exp/mea')/(dataout[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(Path('S0_gaptv/data/exp/img_n')/(dataout[idx][:-4]+'.tiff'),re)

def S0run_test_pnp():
    pool = multiprocessing.Pool()
    PATH = Path('./expdata')
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    MODEL = 'lesti_sst'
    numf = mask.shape[2]
    dataset = []
    datalist = os.listdir(PATH)
    dataout = []
    for idx,name in enumerate(datalist):
        if not 'Lego0019' in name: # small test sets
            print(name)
            continue
        mea = scio.loadmat(PATH/name)['img']
        dataout.append(name)
        dataset.append((mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = pool.starmap(S0run.compressive_model_pnp_exp, dataset)
    if not os.path.exists('S0_gaptv/data/exp'):
        os.mkdir('S0_gaptv/data/exp')
        os.mkdir('S0_gaptv/data/exp/mea')
        os.mkdir('S0_gaptv/data/exp/img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(Path('S0_gaptv/data/exp/mea')/(dataout[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(Path('S0_gaptv/data/exp/img_n')/(dataout[idx][:-4]+'.tiff'),re)

if __name__=='__main__':
    # S0
    #S0run_test()
    # S1
    S1run.test('S0_gaptv/data/exp','S1_denoiser/result/exp','expdata/mask.mat')
    # S2
    S2run.test('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/data/exp/img_n','S0_gaptv/data/exp/','S2_flow_predict/result/exp_wo_s1/')
    S2run.test('/lustre/arce/X_MA/SCI_2.0_python/S1_denoiser/result/exp','S0_gaptv/data/exp/','S2_flow_predict/result/exp_w_s1/')
    # S3
    S3run.test('S2_flow_predict/result/exp_w_s1/re','S0_gaptv/data/exp/', 'S3_spectra_convert/result/exp_ws1')
    S3run.test('S2_flow_predict/result/exp_wo_s1/re','S0_gaptv/data/exp/', 'S3_spectra_convert/result/exp_wos1')
