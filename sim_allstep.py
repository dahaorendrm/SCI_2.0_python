from S0_run import compressive_model_pnp,compressive_model
import S2_test as S2run
import S3_test as S3run
from pathlib import Path
import multiprocessing,threading,queue
import scipy.io as scio
import os
import tifffile
import numpy as np
from skimage import exposure

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







if __name__ == '__main__':
    ## S1
    #pnp_spvicnn_paper('resultpaper/S0/spvi')
    ## S2
    S2run.test('resultpaper/S0/spvi/img_n','resultpaper/S0/spvi','resultpaper/S2/')
    ## S3
    S3run.test('resultpaper/S2/re','resultpaper/S0/spvi/gt', 'resultpaper/S3/result')
