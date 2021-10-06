from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import pickle
import albumentations

class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        #self.data = []
        if os.path.exists(path):
            self.data = []
            for pa in os.listdir(path):
                self.data.append(path+pa)
        else:
            raise FileNotFoundError('path doesnt exist!')



    def __getitem__(self, index):
        #print(index)
        #print(f'Data length :{len(self.data)}')
        #print(f'Data length 2:{len(self.data[0])}')
        #print(self.data)
        training_transformations = albumentations.Compose(
            [
                albumentations.RandomCrop(256, 256),
                albumentations.RandomRotate90(),
                albumentations.HorizontalFlip(),
                albumentations.VerticalFlip(),
            ]
        )

        ground_truth = self.data[index]
        gt = scio.loadmat(ground_truth)['cube']
        #gt = gt.permute(2, 0, 1)
        gts = training_transformations(image=gt)

        return gts

    def __len__(self):

        return len(self.data)
