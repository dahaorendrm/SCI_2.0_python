import numpy as np
import cv2
import scipy.io as scio
import matplotlib.pyplot as plt
#import pickle
import os
from scipy import signal
from pathlib import Path
import tifffile
'''
This code is used to pack a mat file, then used by LESTI model to do the reconstruction

'''
savepath=Path('resultpaper/lesti_compare/mat')

mea = tifffile.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S0/spvi/mea/4D_Lego_24.tiff')
mea = mea/np.amax(mea)*24
mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
mask = mask/np.amax(mask)

led_curve = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/BandsLed.mat')['BandsLed']
led_curve = led_curve[4:-2,:]


nled = led_curve.shape[1]
nf = 3
mask_ = np.expand_dims(np.reshape(mask[:,:,:nled*nf],(256,256,nled,nf)),axis=2)
modul = mask_ * np.expand_dims(led_curve,axis=2)  # Shape:nr,nc,nl,nled,nf
nr=256
nc=256
nl=led_curve.shape[0]
modul = np.sum(modul,axis=3) # Shape:nr,nc,nl,nf
modul = np.swapaxes(modul,2,3)
modul = np.reshape(modul,(nr*nc*nf,nl))
modul = signal.resample(modul,8,axis=1)
modul = np.reshape(modul,(nr,nc,nf,nled))
modul = np.swapaxes(modul,2,3)
led_curve = signal.resample(led_curve,8,axis=0)



scio.savemat(savepath/'4D_Lego_24.mat',{'DATASET_NAME':'4D_Lego_24', 'MODEL':'LESTI',
    'LEDmodul':led_curve, 'mea':mea, 'IND':1,'modul':modul})
