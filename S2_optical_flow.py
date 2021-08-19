from motion import Motion
import pickle
import numpy as np
import os


def outputarray(input,method):
    value = []
    for ind_r in range(input.shape[3]):
        temp = []
        for ind_c in range(input.shape[2]):
            temp.append(method(input[...,ind_c,ind_r]))
        value.append(temp)
    return np.array(value)
    #print('Max and min value of the result is ')
    #print(np.array(value))


path = 'S1_result'
data_list = os.listdir(path)
name = '0000'
for data_name in data_list:
    if name in data_name:
        with np.load(path + '/' + data_name) as data:
            gt_outp = data['gt_outp']
            input = data['input']
            output = data['output']
            gt_orig = data['gt_orig']
            if 'gt_leds' in data.keys():
                gt_leds = data['gt_leds']

re = np.squeeze(output)
#re = np.reshape(re,(256,256,3,int(re.shape[-1]/3)))
re = np.reshape(re,(256,256,8,int(re.shape[-1]/8)))
re = np.moveaxis(re,(0,1,2,3),(-2,-1,-3,-4))
orig_leds = np.squeeze(gt_leds)
orig_leds = orig_leds[...,:24]
#orig_leds = np.squeeze(gt_orig)
print(f'Shape check: re has shape of {re.shape}, orig_leds has shape of {orig_leds.shape}')
#flow = Motion(method='dain_flow',timestep=0.125)
#_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
#logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
#logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
flow = Motion(method='dain_flow2')
re_ledimg_4d,v_psnr,_ = flow.get_motions(re, orig_leds)
v_max = outputarray(re_ledimg_4d,np.amax)
v_min = outputarray(re_ledimg_4d,np.amin)
print('Max and min value of the result is ')
print(v_max)
print(v_min)
print('All psnr is ')
print(v_psnr)
print(f'The psnr of result is {np.mean(np.array(v_psnr))}')
#np.save('temp/dain_result.npy',re_ledimg_4d)
