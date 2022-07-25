import numpy as np
from skimage.measure import block_reduce
import os
from pathlib import Path
import tifffile
import scipy.io as scio
import utils

def converter(path,idxrange,imgsize,cord,scale):
    imgs = []
    for i in range(idxrange[0],idxrange[1]+1):
        data = tifffile.imread(path/ f'Capture{i:03}.tiff')
        #print(data.shape)
        temp = data/1024
        temp = temp[cord[0]:cord[0]+imgsize,cord[1]:cord[1]+imgsize]
        image_rescaled = block_reduce(temp, block_size=(int(1/scale),int(1/scale)), func=np.mean)
        imgs.append(image_rescaled)
    return imgs

if __name__=='__main__':
    ## 20210930data
    if not os.path.exists('expdata'):
        os.mkdir('expdata')
    path = Path('/home/xiao/data/expdata_SCI_2.0/20210930data')
    mask = converter(path,(1,24),1024,(3,1),0.25)
    mask = np.stack(mask,2)
    utils.saveintemp(mask,'expmask')
    scio.savemat('expdata/mask.mat',{'mask':mask})
    imgs = converter(path,(37,40),1024,(3,1),0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/color_checker{idx:04d}.mat',{'img':img})
    imgs = converter(path,(41,80),1024,(3,1),0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/color_checker_motion{idx:04d}.mat',{'img':img})
    imgs = converter(path,(87,116),1024,(3,1),0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/rotor{idx:04d}.mat',{'img':img})
    imgs = converter(path,(317,366),1024,(3,1),0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/Lego{idx:04d}.mat',{'img':img})
    imgs = converter(path,(467,576),1024,(3,1),0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/Mango1_{idx:04d}.mat',{'img':img})
    imgs = converter(path,(567,616),1024,(3,1),0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/Mango2_{idx:04d}.mat',{'img':img})
    ## 20220725data
    savepath = Path('expdata20220725_preprocessed')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    path = Path('/home/xiao/data/expdata_SCI_2.0/20220725data')
    shift = (3,0)
    mask = converter(path/"calihi",(213,236),1024,shift,0.25)
    mask = np.stack(mask,2)
    utils.saveintemp(mask,'expmask')
    scio.savemat(savepath/'mask.mat',{'mask':mask})

    imgs = converter(path/'lego1',(317,356),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'lego1{idx:04d}.mat',{'img':img})

    imgs = converter(path/'lego2_lowexp',(357,396),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'lego2_lowexp{idx:04d}.mat',{'img':img})

    imgs = converter(path/'mario',(798,837),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'mario{idx:04d}.mat',{'img':img})

    imgs = converter(path/'lego_slow',(397,436),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'lego_slow{idx:04d}.mat',{'img':img})

    imgs = converter(path/'lego_arm',(838,877),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'lego_arm{idx:04d}.mat',{'img':img})

    imgs = converter(path/'116fps_toys_high_motion',(1078,1197),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'116fps_toys_high_motion{idx:04d}.mat',{'img':img})

    imgs = converter(path/'text_still',(558,559),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(savepath/f'text_still{idx:04d}.mat',{'img':img})
