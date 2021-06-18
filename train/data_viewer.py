from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import tifffile,pickle
import numpy as np
import matplotlib.pyplot as plt

IND = 650
ground_truth_path = 'data/gt'
feature_path = 'data/feature'

gt_names = os.listdir(ground_truth_path)
feature_names = os.listdir(feature_path)

ground_truth = ground_truth_path + '/' + gt_names[IND]
feature = feature_path + '/' + gt_names[IND]

gt = tifffile.imread(ground_truth)
feat = tifffile.imread(feature)

gt = gt/255.
print(f'Shape of gt is : {gt.shape}')
print(f'Data range of gt is : ({np.amin(gt)}, {np.amax(gt)})')
print(f'Shape of feat is : {feat.shape}')
print(f'Data range of feat is : ({np.amin(feat)}, {np.amax(feat)})')

fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for i in range(0, columns*rows ):
    img = gt[...,i]
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
plt.show()



gts_ = []
ind_c = 0
for ind in range(gt.shape[-1]):
    temp = gt[...,ind_c,ind]
    gts_.append(temp)
    ind_c = ind_c+1 if ind_c<2 else 0
gt_com = np.stack(gts_,-1)
fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for i in range(0, columns*rows ):
    img = gt_com[...,i]
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
plt.show()


mea = feat[...,0]
print(f'Data range of mea is : ({np.amin(mea)}, {np.amax(mea)})')
plt.imshow(mea)


gaptv = feat[...,1:]
print(f'Shape of gt_com is : {gt_com.shape}')
print(f'Data range of gt_com is : ({np.amin(gt_com)}, {np.amax(gt_com)})')
print(f'Shape of gaptv is : {gaptv.shape}')
print(f'Data range of gaptv is : ({np.amin(gaptv)}, {np.amax(gaptv)})')
MSE = np.square(np.subtract(gt_com,gaptv)).mean()
print(f'MSE = {MSE}')



fig = plt.figure(figsize=(8, 8))
for i in range(0, columns*rows ):
    img = gaptv[...,i]
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
plt.show()
