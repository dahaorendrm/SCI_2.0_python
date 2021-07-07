from imgdataset import Imgdataset
import torch
import torch.nn as nn
from networks.chasti_network import CHASTINET
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
import os
from tuils import calculate_psnr,calculate_ssim
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CHASTINET(11,128).to(device)
epoch_ind = 4
model = torch.load('./train/epoch' + "/{}.pth".format(epoch_ind))

def test(test_dataloader):
    MASK = scio.loadmat('./train/lesti_mask.mat')['mask']
    for ind_data, (gts, inputs) in enumerate(test_dataloader):
        with torch.no_grad():
            mea = inputs[...,0] # mea normalize???????????????????????????????????????????? from birnet
            img_ns = inputs[...,1:]
            masks = torch.tensor(MASK[...,:img_ns.size()[-1]]).float()
            masks = masks.repeat(img_ns.size()[0],1,1,1)
            img_n_codes = img_ns*masks
            output = []
            mea = torch.unsqueeze(mea,3)
            for ind in range(img_ns.size()[-1]):
                #gt = gts[...,ind]
                img_n = img_ns[...,ind:ind+1]
                img_n_code_begin = img_n_codes[...,:ind]
                img_n_code_end = img_n_codes[...,ind+1:]
                mask = masks[...,ind:ind+1]
                #print(f'Shape of img_n: {img_n.size()}, img_n_code_begin: {img_n_code_begin.size()}, /nimg_n_code_end: {img_n_code_end.size()}, mask: {mask.size()}, mea: {mea.size()}')
                cat_input = torch.cat((img_n,mea,mask,img_n_code_begin,img_n_code_end),dim=3)
                # Forward pass
                cat_input = np.array(cat_input)
                cat_input = np.moveaxis(cat_input,-1,1)
                cat_input = torch.from_numpy(cat_input).float()
                #print(f'Shap of cat_input is {cat_input.size()}')
                #cat_input = torch.movedim(cat_input,-1,1)
                cat_input = cat_input.to(device)
                output_ = model(cat_input)
                #print(f'Shap of single output is {output_.size()}')
                output.append(output_)
            output = torch.cat(output,1)
            #output = torch.moveaxis(output,0,-1)
            #print(f'output shape is {output.size()}')
            gts_ = []
            ind_c = 0
            for ind in range(gts.size()[-1]):
                temp = gts[...,ind_c,ind]
                gts_.append(temp)
                ind_c = ind_c+1 if ind_c<2 else 0
            gts = torch.stack(gts_,1)/255.
            #gts = torch.moveaxis(gts,0,-1)
            #print('gts shape is' + str(gts.size()))
            output = output.cpu()
            psnr = calculate_psnr(output,gts)
            print(f'Data {ind_data} PSNR is {}.')
            output = output.numpy()
            if not os.path.exists('test/result'):
                os.mkdir('test/result')
            with open("test/result/testresult_{ind_data}.pickle","wb") as f:
                np.save(f, output)


if __name__ == '__main__':
    test_path = 'test/data'
    dataset = Imgdataset(test_path)
    test_dataloader = DataLoader(dataset)
    test(test_dataloader)
