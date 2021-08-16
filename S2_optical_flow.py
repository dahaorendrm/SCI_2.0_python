from motion import Motion
import pickle
import numpy
import os


path = 'test/result'
data_list = os.listdir(path)
name = '0005'
for data in data_list:
    if name in data and 'input' in data:
        with open(path + '/' + data, 'rb') as f:
            inputimgs = numpy.load(f)

    if name in data and 'result' in data:
        with open(path + '/' + data, 'rb') as f:
            outputimgs = numpy.load(f)

    if name in data and 'gt' in data:
        with open(path + '/' + data, 'rb') as f:
            gtimgs = numpy.load(f)

re = outputimgs
orig_leds = gtimgs
#flow = Motion(method='dain_flow',timestep=0.125)
#_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
#logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
#logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
flow = Motion(method='dain_flow2')
re_ledimg_4d,_ = flow.get_motions(re, orig_leds)
