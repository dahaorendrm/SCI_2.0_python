from motion import Motion
import pickle
import numpy as np
import os,utils
from PIL import Image
import itertools
import tifffile
from pathlib import Path


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
        if os.path.isdir(path /'img_n'/ data_name):
            continue
        # load data
        #if name not in data_name:
        #    continue
        save_name = data_name.split('.')[0]
        print(f'Processing data {save_name}')
        input = tifffile.imread(path /'img_n'/ data_name)
        if os.path.exists(path/'gt_led'/data_name):
            gt_orig = tifffile.imread(path/'gt_led'/data_name)
        elif os.path.exists(path/'gt'/data_name):
            gt_orig = tifffile.imread(path/'gt'/data_name)
            gt_orig = gt_orig/255. # all data in gt has 255 max value
        else:
            gt_orig = None
        #print(f'Shape check: data has shape of {data.shape}, orig_leds has shape of {orig_leds.shape}')
        if '4D' in data_name:
            input = reshape_data(input,8)
        else:
            input = reshape_data(input,3)
        re  = process(input,gt_orig)
        # np.save('S2_result/'+save_name+'gt.npy',    re_gt)
        # np.save('S2_result/'+save_name+'input.npy', re_in)
        # np.save('S2_result/'+save_name+'output.npy', re_out)
        # np.save('S2_result/'+save_name+'ref.npy', orig_leds)
        print(f'shape of re is {re.shape}')
        utils.saveintemp(re,save_name)
        utils.saveintemp(gt_orig,'orig'+save_name)
        psnr_re,ssim_re = utils.outputevalarray(re,gt_orig)
        print(f'The avg psnr of gt is {np.mean(psnr_re)}')
        print(f'The avg ssim of gt is {np.mean(ssim_re)}')
        ## Save
        if not os.path.exists('./result/re') or not os.path.exists('./result/eval'):
            os.mkdir('result')
            os.mkdir('result/re')
            os.mkdir('result/eval')
        np.savetxt(Path('result')/'eval'/(save_name+f'_psnr_{np.mean(psnr_re):.4f}.txt'), psnr_re, fmt='%.4f')
        np.savetxt(Path('result')/'eval'/(save_name+f'_ssim_{np.mean(ssim_re):.6f}.txt'), ssim_re, fmt='%.6f')


        tifffile.imwrite(Path('./result/re')/data_name,re)



if __name__ == '__main__':
    main()
