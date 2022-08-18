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
import utils as UTILS


def test_lesti(savepath='resultpaper/lesti_compare', path = './expdata20220723'):
    savepath = Path(savepath)
    pool = multiprocessing.Pool(10)
    PATH = Path(path)
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    mask = mask/np.amax(mask)
    MODEL = 'lesti_sst'
    numf = int(mask.shape[2]/8)
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
    return_crops_data = pool.starmap(compressive_model_gatv4d_exp, dataset)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath/'mea')
        os.mkdir(savepath/'img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(savepath/'mea'/(dataout[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(savepath/'img_n'/(dataout[idx][:-4]+'.tiff'),re)

def test_lesti_sim(savepath='resultpaper/lesti_compare'):
    savepath = Path(savepath)
    gt = tifffile.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S0/spvi/gt/4D_Lego_24.tiff')
    print(gt.shape)
    mea = tifffile.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S0/spvi/mea/4D_Lego_24.tiff')
    mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    mask = mask/np.amax(mask)
    print(mask.shape)
    MODEL = 'lesti_sst'
    numf = 3
    dataset = []
    dataout = []
    dataout.append('4D_Lego_24')
    dataset.append((MODEL,mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = compressive_model_gatv4d_exp(*dataset[0])
    #v_psnr = UTILS.calculate_psnr(return_crops_data[1],gt)
    #v_ssim = UTILS.calculate_ssim(return_crops_data[1],gt)
    #np.savetxt(savepath/'4D_Lego_eval.txt', ['PSNR:',v_psnr, 'SSIM:',v_ssim])
    #print(v_psnr)
    #print(v_ssim)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath/'mea')
        os.mkdir(savepath/'img_n')
    mea = return_crops_data[0]
    re = return_crops_data[1]
    idx=0
    tifffile.imwrite(savepath/'mea'/(dataout[idx]+'.tiff'),mea)
    tifffile.imwrite(savepath/'img_n'/(dataout[idx]+'.tiff'),re)




if __name__ == '__main__':
  #test_lesti(savepath='resultpaper/lesti_compare', path = './expdata20220723')
  test_lesti_sim()
