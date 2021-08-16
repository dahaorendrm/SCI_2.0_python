import scipy.io as scio
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform as skitrans
PATH = 'G:/My Drive/PHD_Research/data/hyper video/hyperspectral-video-33bands'

data_cube = []
for i in range(1,31):
    name_f = os.listdir(PATH+'/f'+str(i))
    name_f = sorted(name_f)
    a_frame = []
    for name in name_f:
        temp = np.asarray(Image.open(PATH+'/f'+str(i)+'/'+name))
        a_frame.append(temp)
    a_frame = np.asarray(a_frame)
    a_frame = np.moveaxis(a_frame,0,-1)
    data_cube.append(a_frame)
data_cube = np.asarray(data_cube)
data_cube = np.moveaxis(data_cube,0,-1)

data_cube.shape
data_cube = data_cube[:,200:200+480,2:,:]
pic_block_down = np.reshape(data_cube,(*data_cube.shape[0:2],np.prod(data_cube.shape[2:])))
pic_block_down = skitrans.resize(pic_block_down,(256,256), # downscale the sub-video
                            anti_aliasing=True)
pic_block_down = np.reshape(pic_block_down,(*pic_block_down.shape[0:2],*data_cube.shape[2:]))
pic_block_down.shape
plt.imshow(pic_block_down[:,:,10,29])
np.amax(pic_block_down)
scio.savemat('data/blocks.mat',{'img':pic_block_down})
a = scio.loadmat('data/4D_Lego.mat')
