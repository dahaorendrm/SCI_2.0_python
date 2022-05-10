import numpy as np
import tifffile as tif
import utils
import os
from colour_system import cs_srgb as ColorS
import cv2
import sys
import scipy.io as scio

def specimg2rgb(data,specrange = (440,680)):
    data_rgb = np.zeros((*data.shape[0:2],3,data.shape[3]))
    for indf in range(data.shape[3]):
        for indi in range(data.shape[0]):
            for indj in range(data.shape[1]):
                data_rgb[indi,indj,:,indf] = ColorS.spec_to_rgb(data[indi,indj,:,indf],specrange)
    data = data_rgb/np.amax(data_rgb)
    return data


def rawvideo(name = '4D_color_checker', nfiles = 4, nframe = 24):

nfiles = 4 # Number of input files
nframe = 24
if not os.path.exists('./video'):
   os.mkdir('./video')
rgblist = []
ledlist = []
mealist = []
for idx in range(nfiles):
    mea =  tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/exp/S0/spvi/mea/'+name+f'{idx:04d}'+'.tiff')
    led =  tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/exp/S0/spvi/img_n/'+name+f'{idx:04d}'+'.tiff')
    img =  tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/exp/S3/result/'+name+f'{idx:04d}'+'.tiff.tiff')
    rgblist.append(specimg2rgb(img))
    mealist.append(mea)
    ledlist.append(led)
rawrergb = np.concatenate(rgblist,3)
rawmea = np.stack(mealist,2)
rawreled = np.concatenate(ledlist,2)
scio.savemat('./video/'+name+'.mat',{'mea':rawmea,'rgb':rawrergb,'led':rawreled})
