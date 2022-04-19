import torch
#import rasterio
import numpy as np
import os
import albumentations
import tifffile
import scipy.io as scio
# These transformations will be passed to our model class

transformations = albumentations.Compose(
    [
        albumentations.RandomCrop(256,256),
        albumentations.RandomRotate90(),
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
    ],
    additional_targets={
        'image1': 'image',
        'image2': 'image'
    }
)

class ImgDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_path, y_path=None, f_trans=True):
        self.feature_path = x_path
        self.data = os.listdir(x_path)
        self.label_path = y_path
        self.f_trans = f_trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = tifffile.imread(self.feature_path+'/'+self.data[idx])
        # Min-max normalization
        min_norm = np.nanmin(feature)
        max_norm = np.nanmax(feature)
        feature = (feature - min_norm) / (max_norm - min_norm)

        # Load label if available - training only
        label = None
        if os.path.exists(self.label_path+'/'+self.data[idx]):
            label = tifffile.imread(self.label_path+'/'+self.data[idx])
            min_norm = np.nanmin(label)
            max_norm = np.nanmax(label)
            label = (label - min_norm) / (max_norm - min_norm)
            if self.f_trans:
                transformed = transformations(image=feature,image1=label)
                feature = transformed['image']
                label = transformed['image1']

        else:
             label = None
        # Prepare sample dictionary
        if feature.ndim == 4:
            feature = np.transpose(feature, [2, 0, 1, 3])
            label = np.transpose(label, [2, 0, 1, 3]) if label is not None else False
        else:
            feature = np.transpose(feature, [2, 0, 1])
            label = np.transpose(label, [2, 0, 1]) if label is not None else False
        sample = {"id": self.data[idx], "feature":feature, "label":label}
        return sample


class TestDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_path,y_path=None):
        self.feature_path = x_path
        self.data = [name for name in os.listdir(x_path) if os.path.isfile(self.feature_path+'/'+name)]
        self.label_path = y_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.load(self.feature_path+'/'+self.data[idx])
        feature = feature['re_out']
        #print(f'feature type : {feature.type()}')
        # Min-max normalization
        min_norm = np.nanmin(feature)
        max_norm = np.nanmax(feature)
        feature = (feature - min_norm) / (max_norm - min_norm)

        # Load label if available - training only
        if self.label_path is not None:
            label = tifffile.imread(self.label_path+'/'+self.data[idx][:4]+'.tiff')
            min_norm = np.nanmin(label)
            max_norm = np.nanmax(label)
            label = (label - min_norm) / (max_norm - min_norm)
        else:
             label = None
        # Prepare sample dictionary
        feature = np.transpose(feature, [2, 0, 1, 3])
        label = np.transpose(label, [2, 0, 1, 3])
        sample = {"id": self.data[idx][:4], "feature":feature, "label":label}
        return sample

class PaperDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_path):
        self.data = [x_path]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = scio.loadmat(self.data[idx])['img']/255.
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        led_curve = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/BandsLed.mat')['BandsLed']
        gt = np.expand_dims(img[:,:,4:-2],axis=3)
        print(gt.shape)
        orig_leds = np.expand_dims(img,axis=[3,4])              # shape:nr, nc, nl,    1,  1 
        led_curve = np.expand_dims(led_curve,axis=2)            # shape:        nl, nled,  1
        img = np.sum(orig_leds * led_curve, axis=2)             # shape:nr, nc,     nled,  1

        # Min-max normalization
        import tifffile as tif
        tif.imwrite('/lustre/arce/X_MA/SCI_2.0_python/S3_spectra_convert/result/re_paper/input.tiff',img)
        min_norm = np.nanmin(img)
        max_norm = np.nanmax(img)
        img = (img - min_norm) / (max_norm - min_norm)

        # Load label if available - training only
        # Prepare sample dictionary
        feature = np.transpose(img, [2, 0, 1, 3])
        label = np.transpose(gt, [2, 0, 1, 3])
        sample = {"id": 'test_paper', "feature":feature, "label":label}
        return sample
