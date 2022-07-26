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
def S0run_test(savepath='resultpaper/exp20220723/S0/gaptv'):
    savepath = Path(savepath)
    pool = multiprocessing.Pool(10)
    PATH = Path('./expdata20220723')
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    MODEL = 'lesti_sst'
    numf = mask.shape[2]
    dataset = []
    datalist = os.listdir(PATH)
    dataout = []
    for idx,name in enumerate(datalist):
        if 'mask' in name: # small test sets
            #print(name)
            continue
        if idx>8:
            continue
        mea = scio.loadmat(PATH/name)['img']
        dataout.append(name)
        dataset.append((MODEL,mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = pool.starmap(S0run.compressive_model_exp, dataset)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath/'mea')
        os.mkdir(savepath/'img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(savepath/'mea'/(dataout[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(savepath/'img_n'/(dataout[idx][:-4]+'.tiff'),re)


def S0run_test_pnp(savepath='resultpaper/exp20220723/S0/spvi'):
    savepath = Path(savepath)
    pool = multiprocessing.Pool(30)
    PATH = Path('./expdata20220723')
    mask = scio.loadmat(PATH/'mask.mat')['mask']
    mask = mask/np.amax(mask)
    MODEL = 'lesti_sst'
    numf = mask.shape[2]
    dataset = []
    datalist = os.listdir(PATH)
    dataout = []
    for idx,name in enumerate(datalist):
        if 'mask' in name: # small test sets
        #    print(name)
            continue
        if idx>3:
            continue
        mea = scio.loadmat(PATH/name)['img']*24*1.4
        dataout.append(name)
        dataset.append((MODEL,mea,mask,numf))
    #S0run.compressive_model_exp(MODEL,mea,mask,numf=16)
    return_crops_data = pool.starmap(S0run.compressive_model_pnp_exp, dataset)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath/'mea')
        os.mkdir(savepath/'img_n')
    for idx,(mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(savepath/'mea'/(dataout[idx][:-4]+'.tiff'),mea)
        tifffile.imwrite(savepath/'img_n'/(dataout[idx][:-4]+'.tiff'),re)

def pnp_spvicnn_paper(savepath='paper/S0/spvi'): # total 30 frames
    savepath = Path(savepath)
    pool = multiprocessing.Pool()
    mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    MODEL = 'lesti_sst'
    #imgs = scio.loadmat('S0_gaptv/blocks.mat')['img']*255
    imgs = scio.loadmat('S0_gaptv/4D_Lego.mat')['img']
    imgs = exposure.adjust_gamma(imgs, 0.5)
    imgs = imgs/np.amax(imgs)*255
    imgs_reverse = np.flip(imgs,3)
    imgs = np.concatenate([imgs,imgs_reverse],3)
    print(f'Input LEGO data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    COMP_FRAME = 24
    crops = []
    crops.append(imgs[:,:,4:-2,0:COMP_FRAME])
    comp_input = [('lesti_sst',crop,mask) for crop in crops]
    return_crops_data = pool.starmap(compressive_model_pnp, comp_input) # contain (original led project, mea, gaptv_result)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath/'mea')
        os.mkdir(savepath/'img_n')
        os.mkdir(savepath/'gt')
        os.mkdir(savepath/'gt_led')
        os.mkdir(savepath/'gt_led_compressed')
    for idx,(orig_leds,orig_ledsfull,mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(Path(savepath/'mea')/('4D_Lego_'+str(re.shape[2])+'.tiff'),mea)
        tifffile.imwrite(Path(savepath/'img_n')/('4D_Lego_'+str(re.shape[2])+'.tiff'),re)
        tifffile.imwrite(Path(savepath/'gt_led_compressed')/('4D_Lego_'+str(re.shape[2])+'.tiff'),orig_leds)
        tifffile.imwrite(Path(savepath/'gt_led')/('4D_Lego_'+str(re.shape[2])+'.tiff'),orig_ledsfull)
        tifffile.imwrite(Path(savepath/'gt')/('4D_Lego_'+str(re.shape[2])+'.tiff'),crops[idx])

if __name__=='__main__':
    # S0
    S0run_test()
    # S2
    #S2run.test('resultpaper/exp/S0/spvi/img_n','resultpaper/exp/S0/spvi','resultpaper/exp/S2/')
    ## S3
    #S3run.test('resultpaper/exp/S2/re','resultpaper/exp/S0/spvi/gt', 'resultpaper/exp/S3/result')
