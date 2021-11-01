import torch
#import rasterio
import numpy as np
import os
import albumentations
import tifffile
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

    def __init__(self, x_path, y_path=None):
        self.feature_path = x_path
        self.data = os.listdir(x_path)
        self.label_path = y_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = tifffile.imread(self.feature_path+'/'+self.data[idx])
        # Min-max normalization
        min_norm = np.nanmin(feature)
        max_norm = np.nanmax(feature)
        feature = (feature - min_norm) / (max_norm - min_norm)

        # Load label if available - training only
        if self.label_path is not None:
            label = tifffile.imread(self.label_path+'/'+self.data[idx])
            min_norm = np.nanmin(label)
            max_norm = np.nanmax(label)
            label = (label - min_norm) / (max_norm - min_norm)
            transformed = transformations(image=feature,image1=label)
            feature = transformed['image']
            label = transformed['image1']

        else:
             label = None
        # Prepare sample dictionary
        feature = np.transpose(feature, [2, 0, 1])
        label = np.transpose(label, [2, 0, 1])
        sample = {"id": self.data[idx], "feature":feature, "label":label}
        return sample




class TestDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_path,y_path=None):
        self.feature_path = x_path
        self.data = [name for name in os.listdir(x_path) if os.path.isfile(name)]
        self.label_path = y_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.load(self.feature_path+'/'+self.data[idx])
        # Min-max normalization
        min_norm = np.nanmin(feature)
        max_norm = np.nanmax(feature)
        feature = (feature - min_norm) / (max_norm - min_norm)

        # Load label if available - training only
        if self.label_path is not None:
            label = tifffile.imread(self.label_path+'/'+self.data[idx])
            min_norm = np.nanmin(label)
            max_norm = np.nanmax(label)
            label = (label - min_norm) / (max_norm - min_norm)
            transformed = transformations(image=feature,image1=label)
            feature = transformed['image']
            label = transformed['image1']

        else:
             label = None
        # Prepare sample dictionary
        feature = np.transpose(feature, [2, 0, 1, 3])
        label = np.transpose(label, [2, 0, 1, 3])
        sample = {"id": self.data[idx], "feature":feature, "label":label}
        return sample
