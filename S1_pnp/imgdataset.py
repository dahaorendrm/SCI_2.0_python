import torch
#import rasterio
import numpy as np
import os
import albumentations
import tifffile
import scipy.io as scio
import scipy
from pathlib import Path
import utils

transformations_rgb = albumentations.Compose(
    [
        #albumentations.RandomCrop(256,256),
        albumentations.RandomRotate90(),
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
    ],
    additional_targets={
        'image1': 'image',
        'image2': 'image',
        'image3': 'image',
        'image4': 'image'
    }
)
transformations_16bands = albumentations.Compose(
    [
        albumentations.RandomCrop(256,256),
        albumentations.RandomRotate90(),
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
    ],
    additional_targets={
        'image1': 'image',
        'image2': 'image',
        'image3': 'image',
        'image4': 'image'
    }
)
np.random.seed(seed=0)
def X2Cube(img,B=[4, 4],skip = [4, 4],bandNumber=16):
    '''
    This function came with the whispers datasets
    '''
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//4, N//4,bandNumber )
    return DataCube

class ImgDataset(torch.utils.data.Dataset):

    def __init__(self, path='./S1_pnp/train_data', raw_data_path=Path('../data/whispers/test'), f_trans='rgb'):
        super(ImgDataset, self).__init__()
        #self.data = []
        path = Path(path)
        self.path = path
        self.f_trans= f_trans
        if os.path.exists(path):
            self.data = os.listdir(path)
        else:
            os.mkdir(path)
            self.prepare_rawdata(raw_data_path,path)
            self.data = os.listdir(path)
        rng_level = np.random.default_rng(12)
        rng_distr = np.random.default_rng(14)
        self.sigma_min, self.sigma_max = 0, 25
        self.sigma_test=self.sigma_max

    def prepare_rawdata(self,raw_data_path,output_path):
        for dataname in os.listdir(raw_data_path):
            dataset = []
            imglist = os.listdir(raw_data_path/dataname/'HSI')
            idx = 0
            temp = []
            while idx < len(imglist):
                img = skio.imread(raw_data_path/dataname/'HSI'/f'{i:04d}.png')
                img = X2Cube(img)
                temp.append(img/511.)
                i += 1
                if len(temp) == 8:
                    img = np.stack(temp,3)

                    print(f'imgs.shape is {imgs.shape}.')
                    data1 = img[...,::2]
                    data1 = utils.selectFrames(data1)
                    dataset.append(data1)
                    data2 = img[...,1::2]
                    data2 = utils.selectFrames(data2)
                    dataset.append(data2)
                    temp = []
            #save
            for idx,data in enumerate(dataset):
                tifffile.imwrite(output_path/'_'.join((dataname,str(idx)+'.tiff')),data)

    def __getitem__(self, index):
        file_name = self.data[index]
        img = tifffile.imread(self.path + '/' + file_name)
        temp = []
        for idx in range(img.shape[2]):
            img[...,idx] = scipy.signal.medfilt2d(img[...,idx], kernel_size=3)
        min_norm = np.amin(img)
        max_norm = np.amax(img)
        img = (img - min_norm) / (max_norm - min_norm)
        img_n = np.copy(img)
        noise_level = rng_level.uniform(self.sigma_min, self.sigma_max)
        img_n += rng_distr.normal(0, noise_level / 255.0, img_n.shape)
        transformed = transformations_16bands(image=img,image1=img_n)
        img = transformed['image']
        img_n = transformed['image1']
        sample = {'id':file_name.split('.')[0], 'img_n':img_n, 'img':img, 'sigma':noise_level}
        return img
    def test(self):
        self.sigma_min = self.sigma_max
    def __len__(self):

        return len(self.data)
