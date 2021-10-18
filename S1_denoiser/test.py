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
model = CHASTINET(4,128,4).to(device)
epoch_ind = 1
model.load_state_dict(torch.load(f'./model_weights/epoch_meaN/{epoch_ind}.pth'))

def normalizer(imgs,masks):
    mask_s = torch.sum(masks,3)
    #mask_s = masksi
    #print(f'normalizer mask_s size is {mask_s.size()}; size of imgs is {imgs.size()}')
    mask_s[mask_s==0] = 1
    imgs = imgs/mask_s
    return imgs

def test(test_dataloader):
    MASK = scio.loadmat('./data/lesti_mask.mat')['mask']
    with torch.no_grad():
        for ind_data, data in enumerate(test_dataloader):
            if len(data) >= 3:
                (_,inputs,gts) = data # for lesti model; use led projection as gts
                CHAN = 8
            else:
                (gts,inputs) = data
                CHAN = 3
            print(f'DATA has {CHAN} channels, gts shape is {gts.size()}, inputs shape is {inputs.size()}')
            mea = inputs[...,0] # mea normalize???????????????????????????????????????????? from birnet
            img_ns = inputs[...,1:]
            masks = torch.tensor(MASK[...,:img_ns.size()[-1]]).float()
            masks = masks.repeat(img_ns.size()[0],1,1,1)
            img_n_codes = img_ns*masks
            gts_ = []
            ind_c = 0
            for ind in range(gts.size()[-1]):
                temp = gts[...,ind_c,ind]
                gts_.append(temp)
                ind_c = ind_c+1 if ind_c<CHAN-1 else 0
            gts = torch.stack(gts_,1)
            gts = gts.numpy()
            gts = np.moveaxis(gts,1,-1)
            output = []
            imgs_n = []
            mea = normalizer(mea,masks)
            mea = torch.unsqueeze(mea,3)
            for ind in range(img_ns.size()[-1]):
                #gt = gts[...,ind]
                img_n = img_ns[...,ind:ind+1]
                img_n_code = torch.sum(img_n_codes[...,:ind],-1) + torch.sum(img_n_codes[...,ind+1:],-1) # sum this two line together, then normalize
                mask_code = torch.sum(masks[...,:ind],-1, keepdim=True) + torch.sum(masks[...,ind+1:],-1, keepdim=True)
                #print(f'size of img_n_code is {img_n_code.size()}; size of mask_code is {mask_code.size()}')
                img_n_code_s = normalizer(img_n_code,mask_code)
                img_n_code_s = torch.unsqueeze(img_n_code_s,-1)
                mask = masks[...,ind:ind+1]
                #print(f'Shape of img_n: {img_n.size()}, img_n_code_begin: {img_n_code_begin.size()}, /nimg_n_code_end: {img_n_code_end.size()}, mask: {mask.size()}, mea: {mea.size()}')
                cat_input = torch.cat((img_n,mea,mask,img_n_code_s),dim=3)
                # Forward pass
                cat_input = np.array(cat_input)
                cat_input = np.moveaxis(cat_input,-1,1)
                cat_input = torch.from_numpy(cat_input).float()
                print(f'test:cat_input size is {cat_input.size()}')
                #print(f'Shap of cat_input is {cat_input.size()}')
                #cat_input = torch.movedim(cat_input,-1,1)
                cat_input = cat_input.to(device)
                output_ = model(cat_input)
                #print(f'Shap of single output is {output_.size()}')
                output.append(output_)
                imgs_n.append(img_n)
                gt = gts[...,ind]
                output_ = output_.cpu().numpy()
                img_n   = img_n.cpu().numpy()
                gt = np.squeeze(gt)
                output_ = np.squeeze(output_)
                img_n = np.squeeze(img_n)
                psnr_in  = calculate_psnr(img_n,gt)
                psnr_out = calculate_psnr(output_,gt - np.min(gt)) / (np.max(gt) - np.min(gt))
                print(f'Data {ind_data} at frame {ind}, input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')
                print(f'PSNR has been improved {(psnr_out-psnr_in)/psnr_in:.2%}')
            output = torch.cat(output,1)
            output = output.cpu()
            imgs_n = torch.cat(imgs_n,3)
            imgs_n = imgs_n.cpu()
            #output = torch.moveaxis(output,0,-1)
            #print(f'output shape is {output.size()}')
            #gts = torch.moveaxis(gts,0,-1)
            #print('gts shape is' + str(gts.size()))
            #print(f'output shape is {output.size()}')
            #print(f'imgs_n shape is {imgs_n.size()}')
            output = output.numpy()
            output = np.moveaxis(output,1,-1)
            imgs_n = imgs_n.numpy()
            # Normalize all data
            gts = gts - np.min(gts)) / (np.max(gts) - np.min(gts)
            imgs_n = imgs_n - np.min(imgs_n)) / (np.max(imgs_n) - np.min(imgs_n)
            gt_orig = data[0]
            gt_orig = gt_orig - np.min(gt_orig)) / (np.max(gt_orig) - np.min(gt_orig)
            if len(data) >= 3:
                gt_leds = data[2]
                gt_leds = gt_leds - np.min(gt_leds)) / (np.max(gt_leds) - np.min(gt_leds)
            psnr_in  = calculate_psnr(imgs_n,gts)
            psnr_out = calculate_psnr(output,gts)
            print(f'Data {ind_data}, input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')
            print(f'This model improves PSNR by {(psnr_out-psnr_in)/psnr_in:.2%}')
            if not os.path.exists('result'):
                os.mkdir('result')
            if len(data) >= 3:
                with open(f"result/test_{ind_data:04d}_spectra_input_psnr={psnr_in:.4f}_result_psnr={psnr_out:.4f}.npz","wb") as f:
                    np.savez(f, gt_outp=gts,input=imgs_n,output=output,gt_orig=gt_orig,gt_leds=gt_leds)
            else:
                with open(f"result/test_{ind_data:04d}_rgb_input_psnr={psnr_in:.4f}_result_psnr={psnr_out:.4f}.npz","wb") as f:
                    np.savez(f, gt_outp=gts,input=imgs_n,output=output,gt_orig=gt_orig)

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
