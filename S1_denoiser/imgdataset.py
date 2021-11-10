import torch
#import rasterio
import numpy as np
import os
import albumentations
import tifffile
import scipy.io as scio

transformations = albumentations.Compose(
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

class ImgDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super(ImgDataset, self).__init__()
        #self.data = []
        if os.path.exists(path):
            self.gt_path = path + '/gt'
            self.mea_path = path + '/mea'
            self.img_n_path = path + '/img_n'
            self.gt_led_path = path + '/gt_led'
            self.mask = scio.loadmat('../S0_gaptv/lesti_mask.mat')['mask']
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
        #noise =
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

        transformed = transformations(image=mea,image1=img_n,
                                image2=mask,image3=gt)
        mea = transformed['image']
        img_n = transformed['image1']
        mask = transformed['image2']
        #mea = np.expand_dims(transformed['image'],0)
        #img_n = np.expand_dims(transformed['image1'],0)
        #mask = np.expand_dims(transformed['image2'],0)
        if gt is not None:
            gt = transformed['image3']
            #gt = np.expand_dims(transformed['image3'],0)
            sample = {'id':file_name.split('.')[0], 'mea':mea,
                'img_n':img_n, 'mask':mask,
                    'label':gt}
            return sample
        sample = {'id':file_name.split('.')[0], 'mea':transformed['image'],
            'img_n':transformed['image1'], 'mask':transformed['image2']}
        return sample

    def __len__(self):

        return len(self.data)
