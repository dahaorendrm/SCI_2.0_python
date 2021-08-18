from motion import Motion
import pickle
import numpy as np
import os


path = 'test/result'
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
re = np.reshape(re,(256,256,3,int(re.shape[-1]/3)))
#re = np.reshape(re,(256,256,8,int(re.shape[-1]/8)))
re = np.moveaxis(re,(0,1,2,3),(-2,-1,-3,-4))
#orig_leds = np.squeeze(gt_leds)
#orig_leds = orig_leds[...,:24]
orig_leds = np.squeeze(gt_orig)
print(f'Shape check: re has shape of {re.shape}, orig_leds has shape of {orig_leds.shape}')
#flow = Motion(method='dain_flow',timestep=0.125)
#_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
#logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
#logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
flow = Motion(method='dain_flow2')
re_ledimg_4d,v_psnr,_ = flow.get_motions(re, orig_leds)
v_max, v_min = []
for ind_r in range(re_ledimg_4d.shape[3]):
    t_max, t_min = []
    for ind_c in range(re_ledimg_4d.shape[2]):
        t_max.append(np.amax(re_ledimg_4d[...,ind_c,ind_r]))
        t_min.append(np.amin(re_ledimg_4d[...,ind_c,ind_r]))
    v_max.append(t_max)
    v_min.append(t_min)
print('Max and min value of the result is ')
print(v_max)
print(v_min)
print(f'The psnr of result is {np.mean(array(v_psnr)):wq
}')
#np.save('temp/dain_result.npy',re_ledimg_4d)
