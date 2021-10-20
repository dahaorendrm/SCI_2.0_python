'''
1. load data from '/lustre/scratch/X_MA/data/ntire2020/NTIRE2020_Train_Spectral', '/lustre/scratch/X_MA/data/ntire2020/NTIRE2020_Validation_Spectral'

2. LED projection (optional) compression and decompression

3. save data in './data/train', './data/validation' npz: 1) 8bands LED image 2) 25 bands spectra image
'''

import scipy.io as scio
import tifffile
import multiprocessing,os

BandsLed = scio.loadmat('data/BandsLed.mat')['BandsLed'] # (spec, iLED)

def LEDprojection(data):
    CUT_BAND = (4,2)
    data = data[:,:,CUT_BAND[0]:-CUT_BAND[1]] # Only 25 bands can be utilized by LEDs
    BandsLed = BandsLed[CUT_BAND[0]:-CUT_BAND[1],:]
    BandsLed = BandsLed[np.newaxis,np.newaxis,:,:]
    data = data[:,:,:,np.newaxis]
    data_led = np.sum(data*BandsLed,2)
    return data_led


def process(path,name,savepath='./S3_spectra_convert/data/train'):
    data = scio.loadmat(path+'/'+name)
    data = data['cube']
    led_data = LEDprojection(data)
    tifffile.imwrite(savepath+f'/label/{name[8:12]}.tiff',data) # label
    tifffile.imwrite(savepath+f'/feature/{name[8:12]}.tiff',led_data) # feature

def processes():
    pool = multiprocessing.Pool()
    name_f = os.listdir('/lustre/scratch/X_MA/data/ntire2020/NTIRE2020_Train_Spectral')
    comp_input = [('/lustre/scratch/X_MA/data/ntire2020/NTIRE2020_Train_Spectral',file) for file in name_f]
    pool.starmap(process, comp_input)

if __name__ == '__main__':
    processes()
