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
def S0run_lego():# total 40 frames
    pool = multiprocessing.Pool()
    mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    MODEL = 'lesti_sst'
    imgs = scio.loadmat('S0_gaptv/4D_Lego.mat')['img']
    imgs_reverse = np.flip(imgs,3)
    imgs = np.concatenate([imgs,imgs_reverse],3)
    print(f'Input LEGO data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for COMP_GROUPS in range(2,10,2):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        COMP_FRAME = COMP_GROUPS * 8
        crops.append(imgs[:,:,4:-2,0:COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(S0run.compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
    if not os.path.exists('S0_gaptv/data/sim'):
        os.mkdir('S0_gaptv/data/sim')
        os.mkdir('S0_gaptv/data/sim/mea')
        os.mkdir('S0_gaptv/data/sim/img_n')
        os.mkdir('S0_gaptv/data/sim/gt')
        os.mkdir('S0_gaptv/data/sim/gt_led')
    for idx,(orig_leds,mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(Path('S0_gaptv/data/sim/mea')/('4D_Lego_CF'+str(re.shape[2])+'.tiff'),mea)
        tifffile.imwrite(Path('S0_gaptv/data/sim/img_n')/('4D_Lego_CF'+str(re.shape[2])+'.tiff'),re)
        tifffile.imwrite(Path('S0_gaptv/data/sim/gt_led')/('4D_Lego_CF'+str(re.shape[2])+'.tiff'),orig_leds)
        tifffile.imwrite(Path('S0_gaptv/data/sim/gt')/('4D_Lego_CF'+str(re.shape[2])+'.tiff'),crops[idx])

def S0run_block(): # total 30 frames
    pool = multiprocessing.Pool()
    mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    MODEL = 'lesti_sst'
    imgs = scio.loadmat('S0_gaptv/blocks.mat')['img']
    imgs_reverse = np.flip(imgs,3)
    imgs = np.concatenate([imgs,imgs_reverse],3)
    print(f'Input LEGO data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for COMP_GROUPS in range(2,6,2):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        COMP_FRAME = COMP_GROUPS * 8
        crops.append(imgs[:,:,4:-2,0:COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(S0run.compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
    if not os.path.exists('S0_gaptv/data/sim'):
        os.mkdir('S0_gaptv/data/sim')
        os.mkdir('S0_gaptv/data/sim/mea')
        os.mkdir('S0_gaptv/data/sim/img_n')
        os.mkdir('S0_gaptv/data/sim/gt')
        os.mkdir('S0_gaptv/data/sim/gt_led')
    for idx,(orig_leds,mea,re) in enumerate(return_crops_data):
        tifffile.imwrite(Path('S0_gaptv/data/sim/mea')/('4D_Blocks_CF'+str(re.shape[2])+'.tiff'),mea)
        tifffile.imwrite(Path('S0_gaptv/data/sim/img_n')/('4D_Blocks_CF'+str(re.shape[2])+'.tiff'),re)
        tifffile.imwrite(Path('S0_gaptv/data/sim/gt_led')/('4D_Blocks_CF'+str(re.shape[2])+'.tiff'),orig_leds)
        tifffile.imwrite(Path('S0_gaptv/data/sim/gt')/('4D_Blocks_CF'+str(re.shape[2])+'.tiff'),crops[idx])



if __name__=='__main__':
    # S0
    #S0run_test()
    #S0run_lego()
    S0run_block()
    #Error()
    # S1
    if not os.path.exists('S1_denoiser/result/sim'):
        os.mkdir('S1_denoiser/result/sim')
    if not os.path.exists('S1_denoiser/result/sim/eval'):
        os.mkdir('S1_denoiser/result/sim/eval')
    #S1run.test('S0_gaptv/data/sim','S1_denoiser/result/sim')
    # S2
    if not os.path.exists('S2_flow_predict/result/sim_wo_s1'):
        os.mkdir('S2_flow_predict/result/sim_wo_s1')
    #S2run.test('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/data/sim/img_n','S0_gaptv/data/sim/','S2_flow_predict/result/sim_wo_s1/')
    if not os.path.exists('S2_flow_predict/result/sim_w_s1'):
        os.mkdir('S2_flow_predict/result/sim_w_s1')
    #S2run.test('/lustre/arce/X_MA/SCI_2.0_python/S1_denoiser/result/sim','S0_gaptv/data/sim/','S2_flow_predict/result/sim_w_s1/')
    # S3
    #S3run.test('S2_flow_predict/result/sim_w_s1/re','S0_gaptv/data/sim/', 'S3_spectra_convert/result/sim_ws1')
    #S3run.test('S2_flow_predict/result/sim_wo_s1/re','S0_gaptv/data/sim/', 'S3_spectra_convert/result/sim_wos1')
