'''
1. load data from '/lustre/scratch/X_MA/data/ntire2020/NTIRE2020_Train_Spectral', '/lustre/scratch/X_MA/data/ntire2020/NTIRE2020_Validation_Spectral'

2. LED projection (optional) compression and decompression

3. save data in './data/train', './data/validation' npz: 1) 8bands LED image 2) 25 bands spectra image
'''

import scipy.io as scio
import tifffile
import multiprocessing,os
import numpy as np
import os
CUT_BAND = (4,2)
BandsLed = scio.loadmat('data/BandsLed.mat')['BandsLed'] # (spec, iLED)
BandsLed = BandsLed[CUT_BAND[0]:-CUT_BAND[1],:]
BandsLed = BandsLed[np.newaxis,np.newaxis,:,:]

def LEDprojection(data):
    data = np.squeeze(data)
    global BandsLed
    global CUT_BAND
    data = data[:,:,CUT_BAND[0]:-CUT_BAND[1]] # Only 25 bands can be utilized by LEDs
    data = data[:,:,:,np.newaxis]
    data_led = np.sum(data*BandsLed,2)
    return data_led


def process(path,name,savepath='./data/train'):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    if not os.path.exists(savepath+'/'+'label'):
        os.mkdir(savepath+'/'+'label')
    if not os.path.exists(savepath+'/'+'feature'):
         os.mkdir(savepath+'/'+'feature')
    data = scio.loadmat(path+'/'+name)
    print(f'file name is {name}')
    data = data['cube']
    led_data = LEDprojection(data)
    tifffile.imwrite(savepath+f'/label/{name[8:12]}.tiff',data) # label
    tifffile.imwrite(savepath+f'/feature/{name[8:12]}.tiff',led_data) # feature

def processes():
    pool = multiprocessing.Pool()
    name_f = os.listdir('../../data/ntire2020/spectral')
    comp_input = [('../../data/ntire2020/spectral',file) for file in name_f]
    pool.starmap(process, comp_input)

if __name__ == '__main__':
    processes()
