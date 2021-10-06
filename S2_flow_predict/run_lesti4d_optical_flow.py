#%matplotlib inline
from func.utils import init_logger
from func import utils
import matplotlib.pyplot as plt
import os
import platform
from func.result import Result
from func.recon_model import ReModel
from func.measurement import Measurement
import dill,pickle
import yaml
import numpy as np
import scipy.io as scio


def load_data(path:str,name:str,dataname:str):
    return scio.loadmat(f'data/{path}/{name}.mat')[dataname]

def show(data):
    plt.figure()
    plt.imshow(data[:,:,10],cmap='gray')
#############################################################


logger = init_logger(__name__)
uname = platform.uname()
logger.info('The machine enviroment: ' + repr(uname))

# lesti4d model
JSON_NAME = 'lesti4d_sst_tv_fdvd'
with open("config/"+JSON_NAME+".yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
logger.info("Read the json file: config/{}.yaml".format(JSON_NAME))
logger.info("Model name is " + config['MODEL'])

## Measurement
data = (
load_data('orig', config['P_FORWARD']['ORIG_DATA'], 'img'),
load_data('mask', config['P_FORWARD']['MASK'],      'mask'),
load_data('led',  config['P_FORWARD']['BandsLed'],  'BandsLed'))

mea = Measurement(model = config['MODEL'], dim = config['DIM'],
                            configs=config['P_FORWARD'], inputs=data)

# Original of compressed frames ###########################
indl = 0
orig_new = []
for indf in range(mea.orig_leds.shape[3]):
    orig_new.append(mea.orig_leds[:,:,indl,indf])
    indl += 1
    if indl >= 8:
        indl = 0
orig_new = np.asarray(orig_new)
orig_new = np.moveaxis(orig_new,0,-1)
lesti_ref = orig_new


indl = 0
orig_new = []
for indg in range(int(mea.orig_leds.shape[3]/8)):
    temp = []
    for indl in range(mea.orig_leds.shape[2]):
        temp.append(mea.orig_leds[:,:,indl,indl + indg*8])
    orig_new.append(temp)
orig_new = np.asarray(orig_new)
dain_input_ref = orig_new
#orig_new = np.moveaxis(orig_new,0,-1)
# ttt.shape
# utils.display_highdimdatacube(ttt)

# Temp - wraped reconstruction frames ####################
orig_ref = []
for indl in range(mea.orig_leds.shape[2]):
    orig_ref.append(mea.orig_leds[:,:,indl,indl+1:indl+8])
orig_ref = np.asarray(orig_ref)
orig_ref = np.moveaxis(orig_ref,0,-2)
# utils.display_highdimdatacube(tt)

# Reconstruction ########################################
P_RECON = config['P_RECONs'][0]
model = ReModel(P_RECON['RECON_MODEL'],P_RECON['RECON_DENOISER'])
model.config(P_RECON)
result = Result(model,mea, modul = mea.mask, orig = lesti_ref)

re = []
for indg in range(int(mea.orig_leds.shape[3]/8)):
    temp = []
    for indl in range(mea.orig_leds.shape[2]):
        temp.append(result[:,:,indg*8+indl])
    re.append(temp)
re = np.asarray(re)

# <codecell> Recon showup  #####################################
fig = utils.display_highdimdatacube(np.moveaxis(re,(0,1),(-1,-2)),transpose=True)
fig.show()
ref = []
for indg in range(int(mea.orig_leds.shape[3]/8)):
    temp = []
    for indl in range(mea.orig_leds.shape[2]):
        temp.append(lesti_ref[:,:,indg*8+indl])
    ref.append(temp)
ref = np.asarray(ref)
fig_ref = utils.display_highdimdatacube(np.moveaxis(ref,(0,1),(-1,-2)),transpose=True)
fig_ref.show()
lesti_ref.shape
re.shape
#init = result
#P_RECON = config['P_RECONs'][1]
#P_RECON['ITERs'] = 4
#model = ReModel(P_RECON['RECON_MODEL'],P_RECON['RECON_DENOISER'])
#model.config(P_RECON)
#result = Result(model,mea, modul = mea.mask, orig = orig_new, init = init)


# <codecell> DAIN  #####################################
from motion import Motion
#flow = Motion(method='dain_flow',timestep=0.125)
#_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
flow = Motion(method='dain_flow2')
re_ledimg_4d,_ = flow.get_motions(re, mea.orig_leds)
#of.show_all_motions()
#print(result.shape)
#print(orig_ref.shape)
#flow = Motion(method='dain_flow',timestep=0.125)
#_ = flow.get_motions(result[:,:,:8],result[:,:,8:16], orig_ref)


# <codecell> # not working
# lesti recon 2 #####################################
with open(r'temp/dainflow2_results.pickle','rb') as f:
    re_ledimg_4d = pickle.load(f)
init = re_ledimg_4d
modul = np.zeros_like(re_ledimg_4d)
indl = 0
for indf in range(modul.shape[3]):
    modul[:,:,indl,indf] = mea.mask[...,indf]
    indl += 1
    if indl >= 8:
        indl = 0
P_RECON = config['P_RECONs'][2]
P_RECON['ITERs'] = 6
model = ReModel(P_RECON['RECON_MODEL'],P_RECON['RECON_DENOISER'])
model.config(P_RECON)
result = Result(model,mea, modul = modul, orig = mea.orig_leds, init = init)
fig = utils.display_highdimdatacube(result[:,:,:,:8],transpose=True)
fig.show()


# <codecell>
from scipy import signal
with open(r'temp/dainflow2_results.pickle','rb') as f:
    re_ledimg_4d = pickle.load(f)
fig = utils.display_highdimdatacube(re_ledimg_4d[:,:,:,:8],transpose=True)
fig.show()
fig_ref = utils.display_highdimdatacube(mea.orig_leds[:,:,:,:8],transpose=True)
fig_ref.show()


led_curve = signal.resample(mea.led_curve,8,axis=0)
orig = signal.resample(mea.orig,8,axis=2)
temp = np.moveaxis(re_ledimg_4d,-1,-2)
shape_ = temp.shape
temp = np.reshape(temp,(np.cumprod(shape_[:3])[2],shape_[3]))
temp = np.linalg.solve(led_curve.transpose(), temp.transpose())
temp = np.reshape(temp.transpose(),shape_)
temp = np.moveaxis(temp,-1,-2)
fig = utils.display_highdimdatacube(temp[:,:,:,:8],transpose=True)
fig.show()
fig_ref = utils.display_highdimdatacube(orig[:,:,:,:8],transpose=True)
fig_ref.show()
