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

if __name__ == '__main__':
    import pandas as pd
    import utils
    import random
    from pathlib import Path
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torch
    import os
    DATA_PATH = Path("training_data")
    train_metadata = pd.read_csv(
        DATA_PATH / "flood-training-metadata.csv", parse_dates=["scene_start"]
    )
    # Sample 3 random floods for validation set
    flood_ids = train_metadata.flood_id.unique().tolist()
    val_flood_ids = random.sample(flood_ids, 3)
    val = train_metadata[train_metadata.flood_id.isin(val_flood_ids)]
    train = train_metadata[~train_metadata.flood_id.isin(val_flood_ids)]
    train_x = utils.get_paths_by_chip(train)
    train_y = train[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

    train_dataset = FloodDataset(
        train_x, train_y
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    sample = next(iter(train_dataloader))
    key_list = ["chip", "nasadem", "extent", "occurrence","recurrence",
    "seasonality", "transitions", "change", 'label']
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    for key,val in sample.items():
        if key in key_list:
            val = torch.squeeze(val)
            if key is 'chip':
                img = torch.moveaxis(val,0,-1)
                img = utils.create_false_color_composite(img.numpy())
                #print(img.shape)
                plt.imsave('temp/'+key+'.png',img)
                #plt.imshow(img)
                #plt.show()
            else:
                plt.imsave('temp/'+key+'.png',val.numpy())
