from S0_run import compressive_model_pnp,compressive_model,compressive_model_gatv4d_exp
import S2_test as S2run
import S3_test as S3run
from pathlib import Path
import multiprocessing,threading,queue
import scipy.io as scio
import os
import tifffile
import numpy as np
from skimage import exposure
import time


def test_lesti(savepath='resultpaper/lesti_compare', path = './expdata20220723'):
    savepath = Path(savepath)
    pool = multiprocessing.Pool(10)
    PATH = Path(path)
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    mask = mask/np.amax(mask)
    MODEL = 'lesti_sst'
    numf = mask.shape[2]/8
    dataset = []
    datalist = os.listdir(PATH)
    dataout = []
    for idx,name in enumerate(datalist):
        if 'mask' in name: # small test sets
        #    print(name)
            continue
        if datalist[idx]!= 'mario0010.mat':
            continue
        mea = scio.loadmat(PATH/name)['img']
        mea = mea/np.amax(mea)*24*1.4
        dataout.append(name)
        dataset.append((MODEL,mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = pool.starmap(S0run.compressive_model_gatv4d_exp, dataset)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath/'mea')
        os.mkdir(savepath/'img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(savepath/'mea'/(dataout[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(savepath/'img_n'/(dataout[idx][:-4]+'.tiff'),re)




if __name__ == '__main__':
  test_lesti(savepath='resultpaper/exp20220723/S0/spvi', path = './expdata20220723')
