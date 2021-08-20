from motion import Motion
import pickle
import numpy as np
import os



def reshape_data(data,step):
    data = np.squeeze(data)
    data = np.reshape(data,(256,256,step,int(data.shape[-1]/step)),order='F')
    data = np.moveaxis(data,(0,1,2,3),(-2,-1,-3,-4))
    return data

def process(data,ref):
    #flow = Motion(method='dain_flow',timestep=0.125)
    #_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
    #logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
    #logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
    flow = Motion(method='dain_flow2')
    re_ledimg_4d,v_psnr,v_ssim = flow.get_motions(data, ref)
    return re_ledimg_4d

path = 'S1_result'
data_list = os.listdir(path)
name = '0000'

for data_name in data_list:
    # load data
    save_name = data_name[5:10]
    with np.load(path + '/' + data_name) as data:
        gt_outp = data['gt_outp']
        input = data['input']
        output = data['output']
        gt_orig = data['gt_orig']
        if 'gt_leds' in data.keys():
            gt_leds = data['gt_leds']

    # process data
    if 'rgb' in data_name:
        save_name = save_name + 'rgb_'
        gt_outp = reshape_data(gt_outp,3)
        input = reshape_data(input,3)
        output = reshape_data(output,3)
        orig_leds = np.squeeze(gt_orig)
    if 'spectra' in data_name:
        save_name = save_name + 'spectra_'
        gt_outp = reshape_data(gt_outp,8)
        input = reshape_data(input,8)
        output = reshape_data(output,8)
        orig_leds = np.squeeze(gt_leds)
        orig_leds = orig_leds[...,:24]
    #print(f'Shape check: data has shape of {data.shape}, orig_leds has shape of {orig_leds.shape}')
    re_gt  = process(gt_outp,orig_leds)
    re_in  = process(input,orig_leds)
    re_out = process(output,orig_leds)
    with open('S2_result/'+save_name+'.npz',"wb") as f:
        np.savez(f, re_gt=re_gt,re_in=re_in, re_out=re_out, ref=orig_leds)
    # np.save('S2_result/'+save_name+'gt.npy',    re_gt)
    # np.save('S2_result/'+save_name+'input.npy', re_in)
    # np.save('S2_result/'+save_name+'output.npy', re_out)
    # np.save('S2_result/'+save_name+'ref.npy', orig_leds)
