from imgdataset import Imgdataset
import torch
import torch.nn as nn
from networks.chasti_network import CHASTINET
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
import os
from utils import calculate_psnr,calculate_ssim
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CHASTINET(11,128).to(device)
epoch_ind = 4
model.load_state_dict(torch.load('./train/epoch' + "/{}.pth".format(epoch_ind)))

def test(test_dataloader):
    MASK = scio.loadmat('./train/lesti_mask.mat')['mask']
    with torch.no_grad():
        for ind_data, data in enumerate(test_dataloader):
            if len(data) >= 3:
                (_,inputs,gts) = data # for lesti model; use led projection as gts
                CHAN = 8
            else:
                (gts,inputs) = data
                CHAN = 3
            mea = inputs[...,0] # mea normalize???????????????????????????????????????????? from birnet
            img_ns = inputs[...,1:]
            masks = torch.tensor(MASK[...,:img_ns.size()[-1]]).float()
            masks = masks.repeat(img_ns.size()[0],1,1,1)
            img_n_codes = img_ns*masks
            output = []
            imgs_n = []
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
                 #print(cat_input.size())
                cat_input = cat_input.to(device)
                output_ = model(cat_input)
                #print(f'Shap of single output is {output_.size()}')
                output.append(output_)
                imgs_n.append(img_n)
            output = torch.cat(output,1)
            output = output.cpu()
            imgs_n = torch.cat(imgs_n,3)
            imgs_n = imgs_n.cpu()
            #output = torch.moveaxis(output,0,-1)
            #print(f'output shape is {output.size()}')
            gts_ = []
            ind_c = 0
            for ind in range(gts.size()[-1]):
                temp = gts[...,ind_c,ind]
                gts_.append(temp)
                ind_c = ind_c+1 if ind_c<CHAN-1 else 0
            gts = torch.stack(gts_,1)/255.
            #gts = torch.moveaxis(gts,0,-1)
            #print('gts shape is' + str(gts.size()))
            #print(f'output shape is {output.size()}')
            #print(f'imgs_n shape is {imgs_n.size()}')
            output = output.numpy()
            output = np.moveaxis(output,1,-1)
            imgs_n = imgs_n.numpy()
            gts = gts.numpy()
            gts = np.moveaxis(gts,1,-1)
            psnr_in  = calculate_psnr(imgs_n*255,gts*255)
            psnr_out = calculate_psnr(output*255,gts*255)
            print(f'Data {ind_data}, input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')
            print(f'This model improves PSNR by {(psnr_out-psnr_in)/psnr_in:.2%}')
            if not os.path.exists('test/result'):
                os.mkdir('test/result')
            with open(f"test/result/test_{ind_data:04d}_input_psnr={psnr_in:.4f}.npy","wb") as f:
                np.save(f, imgs_n)
            with open(f"test/result/test_{ind_data:04d}_result_psnr={psnr_out:.4f}.npy","wb") as f:
                np.save(f, output)
            with open(f"test/result/test_{ind_data:04d}_gt.npy","wb") as f:
                np.save(f, gts)

def validation_func():
    test_path = 'data/data/validation'
    dataset = Imgdataset(test_path)
    test_dataloader = DataLoader(dataset)
    test(test_dataloader)

def test_func():
    test_path = 'data/data/test'
    dataset = Imgdataset(test_path)
    test_dataloader = DataLoader(dataset)
    test(test_dataloader)

if __name__ == '__main__':
    test_func()
