import torch
#import rasterio
import numpy as np
import os
import albumentations
import tifffile
import scipy.io as scio

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
        albumentations.RandomSizedCrop([60,256],256,256),
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

class ImgDataset(torch.utils.data.Dataset):

    def __init__(self, path, mask_path='../S0_gaptv/lesti_mask.mat', f_trans='rgb'):
        super(ImgDataset, self).__init__()
        #self.data = []
        self.f_trans= f_trans
        if os.path.exists(path):
            self.gt_path = path + '/gt'
            self.mea_path = path + '/mea'
            self.img_n_path = path + '/img_n'
            self.gt_led_path = path + '/gt_led'
            self.mask = scio.loadmat(mask_path)['mask']
            self.data = os.listdir(self.mea_path)
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        file_name = self.data[index]
        mea = tifffile.imread(self.mea_path + '/' + file_name)
        min_norm = np.amin(mea)
        max_norm = np.amax(mea)
        mea = (mea - min_norm) / (max_norm - min_norm)
        img_n = tifffile.imread(self.img_n_path + '/' + file_name)
        min_norm = 0
        max_norm = np.amax(img_n)
        img_n = (img_n - min_norm) / (max_norm - min_norm)
        mask = self.mask[:,:,:img_n.shape[2]]
        oth_n = np.empty_like(img_n)
        temp = img_n*mask
        for i in range(oth_n.shape[-1]):
            oth_n[...,i] = np.mean(np.concatenate((temp[...,:i],temp[...,i:]),2), 2)
        min_norm = np.amin(oth_n)
        max_norm = np.amax(oth_n)
        oth_n = (oth_n - min_norm) / (max_norm - min_norm)

        if os.path.exists(self.gt_led_path + '/' + file_name):
            gt = tifffile.imread(self.gt_led_path + '/' + file_name)
            min_norm = np.amin(gt)
            max_norm = np.amax(gt)
            gt = (gt - min_norm) / (max_norm - min_norm)
        elif os.path.exists(self.gt_path + '/' + file_name):
            gt = tifffile.imread(self.gt_path + '/' + file_name)
            min_norm = np.amin(gt)
            max_norm = np.amax(gt)
            gt = (gt - min_norm) / (max_norm - min_norm)
        else:
            gt = None
        if self.f_trans=='rgb':
            transformed = transformations_rgb(image=mea,image1=img_n,
                                image2=mask,image3=gt,image4=oth_n)
            mea = transformed['image']
            img_n = transformed['image1']
            mask = transformed['image2']
            oth_n = transformed['image4']
            gt = transformed['image3']
        if self.f_trans=='16bands':
            transformed = transformations_16bands(image=mea,image1=img_n,
                                image2=mask,image3=gt,image4=oth_n)
            mea = transformed['image']
            img_n = transformed['image1']
            mask = transformed['image2']
            oth_n = transformed['image4']
            gt = transformed['image3']

        if gt is not None:
            sample = {'id':file_name.split('.')[0], 'mea':mea,
                'img_n':img_n, 'mask':mask,'oth_n':oth_n,
                    'label':gt}
            return sample
        sample = {'id':file_name.split('.')[0], 'mea':mea,
            'img_n':img_n, 'mask':mask, 'oth_n':oth_n}
        return sample

    def __len__(self):

        return len(self.data)
