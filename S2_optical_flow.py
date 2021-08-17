from motion import Motion
import pickle
import numpy
import os


path = 'test/result'
data_list = os.listdir(path)
name = '0003'
for data_name in data_list:
    if name in data_name:
        with numpy.load(path + '/' + data_name) as data:
            gt_outp = data['gt_outp']
            input = data['input']
            output = data['output']
            gt_orig = data['gt_orig']
            if 'gt_leds' in data.keys():
                gt_leds = data['gt_leds']

re = np.squeeze(output)
orig_leds = gt_leds
#flow = Motion(method='dain_flow',timestep=0.125)
#_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
#logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
#logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
flow = Motion(method='dain_flow2')
re_ledimg_4d,_ = flow.get_motions(re, orig_leds)
