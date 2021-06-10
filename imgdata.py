from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import tifffile,pickle



class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            groung_truth_path = path + '/gt'
            measurement_path = path + '/input'

            if os.path.exists(groung_truth_path) and os.path.exists(measurement_path):
                groung_truth = os.listdir(groung_truth_path)
                measurement = os.listdir(measurement_path)
                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                              'measurement': measurement_path + '/' + measurement[i]} for i in range(len(groung_truth))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        groung_truth, measurement = self.data[index]["groung_truth"], self.data[index]["measurement"]

        gt = tifffile.imread(groung_truth)
        meas = tifffile.imread(measurement)

        #gt = torch.from_numpy(gt / 255)
        #meas = torch.from_numpy(meas / 255)

        #gt = gt.permute(2, 0, 1)

        return gt, meas

    def __len__(self):

        return len(self.data)
