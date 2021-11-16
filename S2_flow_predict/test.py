from motion import Motion
import pickle
import numpy as np
import os,utils
from PIL import Image
import itertools
import tifffile
from pathlib import Path

def outputevalarray(data,ref):
    v_psnr = []
    v_ssim = []
    for indr in range(data.shape[3]):
        temp_psnr = []
        temp_ssim = []
        for indc in range(data.shape[2]):
            psnr_ = utils.calculate_psnr(
                       ref[:,:,indc,indr],data[:,:,indc,indr])
            ssim_ = utils.calculate_ssim(
                       ref[:,:,indc,indr],data[:,:,indc,indr])
            temp_ssim.append(ssim_)
            temp_psnr.append(psnr_)
        v_psnr.append(temp_psnr)
        v_ssim.append(temp_ssim)
    return np.array(v_psnr),np.array(v_ssim)

def reshape_data(data,step):
    data = np.squeeze(data)
    data = np.reshape(data,(256,256,step,int(data.shape[-1]/step)),order='F')
    data = np.moveaxis(data,(0,1,2,3),(-2,-1,-3,-4))
    return data

def process(data,ref=None):
    #flow = Motion(method='dain_flow',timestep=0.125)
    #_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
    #logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
    #logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
    flow = Motion(method='dain_flow2')
    re_ledimg_4d,v_psnr,v_ssim = flow.get_motions(data, ref)
    return re_ledimg_4d

def main():
    path = Path('../S0_gaptv/data/test')
    data_list = os.listdir(path/'img_n')
    name = '0000'

    for data_name in data_list:
        if os.path.isdir(path / data_name):
            continue
        # load data
        #if name not in data_name:
        #    continue
        save_name = data_name.split('.')[0]
        input = tifffile.imread(path / data_name)
        if os.path.exist(path/'gt_leds'/data_name):
            gt_orig = tifffile.imread(path/'gt_leds'/data_name)
        elif os.path.exist(path/'gt'/data_name):
            gt_orig = tifffile.imread(path/'gt'/data_name)
        else:
            gt_orig = None
        #print(f'Shape check: data has shape of {data.shape}, orig_leds has shape of {orig_leds.shape}')
        re  = process(input,gt_orig)
        # np.save('S2_result/'+save_name+'gt.npy',    re_gt)
        # np.save('S2_result/'+save_name+'input.npy', re_in)
        # np.save('S2_result/'+save_name+'output.npy', re_out)
        # np.save('S2_result/'+save_name+'ref.npy', orig_leds)
        print(f'shape of re is {re.shape}')
        utils.saveintemp(re,'result')
        psnr_gt,ssim_gt = outputevalarray(re,gt_orig)
        print(f'The avg psnr of gt is {np.mean(psnr_gt)}')
        print(f'The avg ssim of gt is {np.mean(ssim_gt)}')
        ## Save
        tiff.imwrite(Path('./result/re')/data_name)



if __name__ == '__main__':
    main()
