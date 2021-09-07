
'''
step 1 import data
step 2 compressive_model in: 25 bands out:8 led bands, a snapshot measurement
step 3 gap-tv
step 4 chaisti improve
step 5 spectral convertion model
    5.1 setup model
    5.2 load weight
    5.3 run model

'''
from S3_imgdataset import Imgdataset
import torch
from torch.utils.data import DataLoader
# import data
path = '/lustre/arce/X_MA/data/ntire2020/NTIRE2020_Train_Spectral'
dataset = Imgdataset(path)
# These transformations will be passed to our model class

batch_size = 2
train_dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
data = train_dataloader[0]
