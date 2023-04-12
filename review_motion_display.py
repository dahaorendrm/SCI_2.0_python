from PIL import Image
import cv2 as cv
import pickle
import numpy as np


DATALABEL ='5-13'
DATAIDX = 1


file = open('temp/dainflow2_motion_maps.pickle','rb')
data = pickle.load(file)
imga = np.squeeze(data[DATALABEL][DATAIDX][0])
imga = imga/np.amax(imga)*255
imgb = np.squeeze(data[DATALABEL][DATAIDX][1])
imgb = imgb/np.amax(imgb)*255


magnitude, angle = cv.cartToPolar(imga[0,...], imga[1,...])
mask = np.zeros((256,256,3))
mask[..., 0] = angle * 180 / np.pi / 2
mask[..., 1] = 0
mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
rgb = cv.cvtColor(mask.astype(np.float32), cv.COLOR_HSV2BGR)
rgbimg = Image.fromarray(rgb.astype(np.uint8),'RGB')
rgbimg.save('temp/flowbw'+DATALABEL+'_a.jpg')


magnitude, angle = cv.cartToPolar(imgb[0,...], imgb[1,...])
mask = np.zeros((256,256,3))
mask[..., 0] = angle * 180 / np.pi / 2
mask[..., 1] = 0
mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
rgb = cv.cvtColor(mask.astype(np.float32), cv.COLOR_HSV2BGR)
rgbimg = Image.fromarray(rgb.astype(np.uint8),'RGB')
rgbimg.save('temp/flowbw'+DATALABEL+'_b.jpg')

###########################################

DATALABEL ='2-0'
DATAIDX = 4


file = open('temp/dainflow2_motion_maps.pickle','rb')
data = pickle.load(file)
imga = np.squeeze(data[DATALABEL][DATAIDX])
imga = imga/np.amax(imga)*255


magnitude, angle = cv.cartToPolar(imga[0,...], imga[1,...])
mask = np.zeros((256,256,3))
mask[..., 0] = angle * 180 / np.pi / 2
mask[..., 1] = 0
mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
rgb = cv.cvtColor(mask.astype(np.float32), cv.COLOR_HSV2BGR)
rgbimg = Image.fromarray(rgb.astype(np.uint8),'RGB')
rgbimg.save('temp/flowbw'+DATALABEL+'_a.jpg')


######################################################################
DATALABEL ='11-15'



file = open('temp/dainflow2_motion_maps.pickle','rb')
data = pickle.load(file)
imga = np.squeeze(data[DATALABEL])
imga = imga/np.amax(imga)*255


magnitude, angle = cv.cartToPolar(imga[0,...], imga[1,...])
mask = np.zeros((256,256,3))
mask[..., 0] = angle * 180 / np.pi / 2
mask[..., 1] = 0
mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
rgb = cv.cvtColor(mask.astype(np.float32), cv.COLOR_HSV2BGR)
rgbimg = Image.fromarray(rgb.astype(np.uint8),'RGB')
rgbimg.save('temp/flowbw'+DATALABEL+'_a.jpg')



file = open('temp/dainflow2_results.pickle','rb')
data = pickle.load(file)

