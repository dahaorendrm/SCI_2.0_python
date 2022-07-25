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
    if not os.path.exists('expdata20220725_preprocessed'):
        os.mkdir('expdata20220725_preprocessed')
    path = Path('/home/xiao/data/expdata_SCI_2.0/20220725data')
    shift = (3,0)
    mask = converter(path,(1,24),1024,shift,0.25)
    mask = np.stack(mask,2)
    utils.saveintemp(mask,'expmask')
    scio.savemat('expdata/mask.mat',{'mask':mask})
    imgs = converter(path,(37,40),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/color_checker{idx:04d}.mat',{'img':img})
    imgs = converter(path,(41,80),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/color_checker_motion{idx:04d}.mat',{'img':img})
    imgs = converter(path,(87,116),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/rotor{idx:04d}.mat',{'img':img})
    imgs = converter(path,(317,366),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/Lego{idx:04d}.mat',{'img':img})
    imgs = converter(path,(467,576),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/Mango1_{idx:04d}.mat',{'img':img})
    imgs = converter(path,(567,616),1024,shift,0.25)
    for idx,img in enumerate(imgs):
        scio.savemat(f'expdata/Mango2_{idx:04d}.mat',{'img':img})
