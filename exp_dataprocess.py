import numpy as np
from skimage.measure import block_reduce
import os
for pathlib import Path
import tifffile
import scipy.io as scio
import utils

def converter(path,idxrange,imgsize,cord,scale):
    data = []
    for i in range(idxrange[0],idxrange[1]):
        data = tifffile.imread(path/ f'Capture{i:3d}.tiff')
    imgs = []
    for i in range(len(data)):
        temp = data[i]
        temp = temp[cord[0]:cord[0]+imgsize,cord[1]:cord[1]+imgsize]
        image_rescaled = block_reduce(temp, block_size=(1/scale,1/scale), func=np.mean)
        imgs.append(temp)
    return imgs

if __name__=='__main__':
    os.mkdir('expdata')
    path = Path('/lustre/arce/X_MA/data/expdata_SCI_2.0/20210930data')
    mask = converter(path,(1,24),1024,(1,2),0.25)
    mask = np.stack(mask,3)
    utils.saveintemp(mask,'expmask')
    # scio.savemat('expdata/mask.mat',{'mask':mask})
    # imgs = converter(path,(37,40),1024,(1,2),0.25)
    # for idx,img in enumerate(imgs):
    #     scio.savemat(f'expdata/color_checker{idx:4d}.mat',{'img':img})
