import torch
#import rasterio
import numpy as np
import os
import albumentations
import tifffile

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
            self.mask = scio.loadmat('./data/lesti_mask.mat')['mask']
            self.data = os.listdir(mea_path)
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        file_name = self.data[index]
        mea = tifffile.imread(self.mea_path + '/' + file_name)
        img_n = tifffile.imread(self.img_n_path + '/' + file_name)
        mask = self.mask[:,:,:img_n.shape[2]]
        #noise =
        if os.path.exists(self.gt_led_path + '/' + file_name):
            gt = tifffile.imread(self.gt_led_path + '/' + file_name)

        elif os.path.exists(self.gt_path + '/' + file_name):
            gt = tifffile.imread(self.gt_path + '/' + file_name)
        else:
            gt = None

        transformed = transformations(image=mea,image1=img_n,
                                image2=mask,image3=gt)
        if gt is not None:
            sample = {'id':file_name.split('.')[0], 'mea':transformed['image'],
                'img_n':transformed['image1'], 'mask':transformed['image2'],
                    'gt':transformed['image3']}
        else:
            sample = {'id':file_name.split('.')[0], 'mea':transformed['image'],
                'img_n':transformed['image1'], 'mask':transformed['image2']}
        return sample

    def __len__(self):

        return len(self.data)
