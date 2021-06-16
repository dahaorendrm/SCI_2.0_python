from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import tifffile,pickle

MAXLEN = 200

class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        #self.data = []
        if os.path.exists(path):
            ground_truth_path = path + '/gt'
            measurement_path = path + '/feature'
            if os.path.exists(ground_truth_path) and os.path.exists(measurement_path):
                feature_names = os.listdir(ground_truth_path)
                #mea_names = os.listdir(measurement_path)
                #self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                #              'measurement': measurement_path + '/' + measurement[i]} for i in range(len(groung_truth))]
                self.data = [{'ground_truth': ground_truth_path + '/' + feature_names[i],
                     'measurement': measurement_path + '/' + feature_names[i]} for i in range(MAXLEN)]
                #print(f"test: {self.data[10]['ground_truth']}")
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):
        #print(index)
        #print(f'Data length :{len(self.data)}')
        #print(f'Data length 2:{len(self.data[0])}')
        #print(self.data)
        ground_truth = self.data[index]['ground_truth'] 
        measurement = self.data[index]['measurement']
        
        #print(f'Path of gt is {ground_truth}')
        gt = tifffile.imread(ground_truth)
        meas = tifffile.imread(measurement)
        gt = torch.from_numpy(gt).float()
        meas = torch.from_numpy(meas).float()
        #gt = torch.from_numpy(gt / 255)
        #meas = torch.from_numpy(meas / 255)

        #gt = gt.permute(2, 0, 1)

        return gt, meas

    def __len__(self):

        return len(self.data)
