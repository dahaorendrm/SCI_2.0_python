import pickle
import datetime
import math
import os
import json
import logging
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from . import utils
from .denoiser import denoiser
from .utils import load_data,timer

logger = utils.init_logger(__name__)


class ReModel:
    def __init__(self,modelname:'gap/admm',denoisername:'denoiser type'='tv'):
        self.modelname = modelname.lower()
        self.denoisertype = denoisername.lower()
        self.configp = {}
        self.config_de_p = {}
        self.configp["sigmas"] = 1
        self.configp["ACC"] = 1
        self.configp["lambda"] = 1
        self.configp["ITERs"] = 125
        self.configp["ASSESE"] = 1
        self.configp["gamma"] = 0.02
        logger.info('%s model Initialized with %s denoiser.' %(self.modelname,self.denoisertype))

    def config(self,dic):
        for k,v in dic.items():
            if k in self.configp.keys():
                self.configp[k] = dic[k]
                logger.info('Parameter %s is set to: %s' %(k,self.configp[k]))
            if k == 'P_DENOISE':
                self.config_denoise(dic[k])

    def config_denoise(self,dic={}):
        for k,v in dic.items():
            self.config_de_p[k] = dic[k]
            logger.info('Denoiser parameter %s is set to: %s' %(k,self.config_de_p[k]))
        if self.denoisertype == 'tv':
            self.denoiser = lambda x, **kwargs: denoiser.tv_denoiser( x,
                        self.config_de_p["TV_WEIGHT"],self.config_de_p["TV_ITER"])
            return
        if self.denoisertype == 'tv_chambolle':
            self.denoiser = lambda x, **kwargs: denoise_tv_chambolle( x,
                        weight = self.config_de_p["TV_WEIGHT"],
                        n_iter_max = self.config_de_p["TV_ITER"],
                        multichannel=True)
            return
        if 'hsi' in self.denoisertype:
            model,device = denoiser.hsicnn_denoiser_config()
            self.denoiser = lambda x, sigma, it: denoiser.hsicnn_denoiser( x,
                it = it, model = model, device = device, tv_weight = sigma,
                ch_sigma = self.config_de_p['ch_sigma'], tv_iter = self.config_de_p['tv_iter'],
                it_list = self.config_de_p['it_list'])
            return
        if 'ffd' in self.denoisertype:
            model,device = denoiser.ffdnet_denoiser_config()
            self.denoiser = lambda x, sigma, it: denoiser.ffdnet_denoiser( x,
                model = model, device = device, sigma = sigma)
            logger.debug('ffd denoiser is added')
            return
        if 'fdvd' in self.denoisertype:
            model,device = denoiser.fastdvdnet_denoiser_config()
            self.denoiser = lambda x, sigma, it: denoiser.fastdvdnet_denoiser( x,
                model = model, device = device, sigma = sigma)
            logger.debug('fastdvd denoiser is added')
            return
        raise error('No denoiser added.')

    def calculate(self,mea,modul=None,orig=None,init=None):
        # Assign shifter
        # Call model
        if modul is None: modul =  mea.modul
        if orig is None : orig = mea.orig
        if self.modelname == 'gap':
            data,psnr,ssim = gap_model(mea.mea, modul, self.denoiser, mea.afunc, mea.atfunc,
                                init_v=init,
                                orig=orig,
                                sigmas = self.configp["sigmas"],
                                lambda_ = self.configp["lambda"],
                                ITERs = self.configp["ITERs"],
                                ACC=self.configp["ACC"],
                                ASSESE=self.configp["ASSESE"])
        if self.modelname == 'admm':
            data,psnr,ssim = admm_model(mea.mea, modul, self.denoiser, mea.afunc, mea.atfunc,
                                init_v=init,
                                orig=orig,
                                sigmas = self.configp["sigmas"],
                                lambda_ = self.configp["lambda"],
                                ITERs = self.configp["ITERs"],
                                ASSESE=self.configp["ASSESE"],
                                gamma=self.configp["gamma"])
        return data,psnr,ssim

############################gap/admm#############################

@timer(logger)
def gap_model(yy, phi, denoiser, Afunc, Atfunc,
                orig=None, init_v=None,
                lambda_=1, ITERs=100, ACC=1, ASSESE=1,
                sigmas= 130/255):
    logger.info('Gap model running...')
    if init_v is None:
        logger.info('No initialization.')
        init_v = Atfunc(yy,phi)
    vv = init_v
    y_ = np.zeros(yy.shape)
    psnr_record = []
    ssim_record = []
    phisum = np.sum(phi**2,tuple([i for i in range(2,init_v.ndim)]))
    phisum[phisum==0]=1
    gap_func = lambda vv, yy, yb : vv + lambda_ * Atfunc((yy-yb)/phisum, phi)
    # Iteration start
    IT_cum = np.cumsum(ITERs,dtype=int)
    for it in range(np.sum(ITERs)):
        if not isinstance(sigmas, int) and not isinstance(sigmas, float):
            for ind,val in enumerate(IT_cum):
                if it <= val:
                    sigma = sigmas[ind]
                    break
        else:
            sigma=sigmas
        yb = Afunc(vv,phi)
        # [1.1] Gap projection
        if ACC:
            y_ = y_ + (yy-yb) # iterative update y
            vv = gap_func(vv, y_, yb)
        else:
            vv = gap_func(vv, yy, yb)
        # [1.2] Denoising
        #if SHIFTER:
            #vv = SHIFTER(vv, reverse = True)
        vv = denoiser(vv,sigma=sigma,it=it)
        #if SHIFTER:
            #vv = SHIFTER(vv)
        # shift
        # [1.3] Evaluation
        if ASSESE and orig is not None and np.mod(it,ASSESE) == 0:
            #print(f'shape of vv is {vv.shape}, shape of orig is {orig.shape}')
            temp_psnr =utils.calculate_psnr(vv*255,orig*255)
            temp_ssim =utils.calculate_ssim(vv*255,orig*255)
            psnr_record.append(temp_psnr)
            ssim_record.append(temp_ssim)
            #logger.debug("PSNR is "+ str(temp_psnr))
            #logger.debug("SSIM is "+ str(temp_ssim))
            logger.info(
                "Iteration {0:3d}, Sigma = {1:2.2f}, PSNR = {2:2.2f} dB, SSIM = {3:.4f}.".
                format(it+1,sigma,temp_psnr,temp_ssim))
                #print('Iteration {0:3d}, PSNR = {1:2.2f} dB'.format(it,  psnr_record[it]))
            if it == np.sum(ITERs) - 1:
                np.save('temp/psnr',psnr_record)
                np.save('temp/ssim',ssim_record)
        else:
            logger.info(f'Iteration {it+1:3d}, Sigma = {sigma:2.2f}.')
    #if SHIFTER:
    #    vv = SHIFTER(vv, reverse = True)
    return vv,psnr_record,ssim_record

@timer(logger)
def admm_model(yy:'measurement', phi:'modulations', denoiser, Afunc, Atfunc,
                orig=None, init_v=None,
                gamma=0.01,lambda_=1, ITERs=100, ASSESE=1,
                sigmas= 10/255):
    logger.info('Admm model running...')
    if init_v is None:
        init_v = Atfunc(yy,phi)
    #y_ = np.zeros_like(yy)
    bb = np.zeros_like(init_v)
    psnr_record = []
    ssim_record = []
    phisum = np.sum(phi,2)
    phisum[phisum==0]=1
    admm_func = lambda theta, y_,  bb: (theta+bb) + lambda_ * (Atfunc((yy-y_) / (phisum+gamma),phi))
    # Iteration start
    vv = init_v
    theta = init_v
    IT_cum = np.cumsum(ITERs,dtype=int)
    for it in range(np.sum(ITERs)):
        if not isinstance(sigmas, int) and not isinstance(sigmas, float):
            for ind,val in enumerate(IT_cum):
                if it <= val:
                    sigma = sigmas[ind]
                    break
        else:
            sigma=sigmas
        # [1] Start iteration
        # [1.1]  input theta return vv
        yb = Afunc(theta+bb,phi)
        vv = admm_func(theta, yb, bb)
        # [1.2] Denoising / Input : vv return : theta
        theta = vv-bb
        theta = denoiser(theta,sigma=sigma)
        bb = bb - (vv-theta)
        # [1.3] Evaluation
        if ASSESE and orig is not None and np.mod(it,ASSESE) == 0:
            temp_psnr =utils.calculate_psnr(vv*255,orig*255)
            temp_ssim =utils.calculate_ssim(vv*255,orig*255)
            psnr_record.append(temp_psnr)
            ssim_record.append(temp_ssim)
            logger.info(
                "Iteration {0:3d}, Sigma = {1:2.2f}, PSNR = {2:2.2f} dB, SSIM = {3:.4f}.".
                format(it+1,sigma,temp_psnr,temp_ssim))
            if it == np.sum(ITERs) - 1:
                np.save('temp/psnr',psnr_record)
                np.save('temp/ssim',ssim_record)
        else:
            logger.info(f'Iteration {it+1:3d}, Sigma = {sigma:2.2f}.')

    return theta,psnr_record,ssim_record

# def Afunc(x, Phi):
#     '''
#     Forward model of snapshot compressive imaging (SCI), where multiple coded
#     frames are collapsed into a snapshot measurement.
#     '''
#     return np.sum(x*Phi, axis=2)  # element-wise product
#
# def Atfunc(y, Phi):
#     '''
#     Tanspose of the forward model.
#     '''
#     return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)



def recon(mea=None, JSON_NAME = "default_config"):
    from func.result import Result
    with open("config/"+JSON_NAME+".json") as f:
        config = json.load(f)
    logger.info("Read the json file: config/{}.json".format(JSON_NAME))
    model = MyModel(config['RECON_MODEL'],config['RECON_DENOISER'])
    model.config(config['P_RECON'])
    model.config_denoise(config['P_DENOISE'])
    result = Result(model,mea)
    result.save()
    return result


if __name__ == "__main__":
    with open(r'dataset/2021-02-15_113403_Model=CASSI_Input='+
    '3D_DOLL_Mask=white_1_Shift=0.pickle','rb') as f:
        mea = pickle.load(f)
    result = recon(mea)
    result.shape
    plt.imshow(result[:,:,10],cmap='gray',norm=plt.Normalize(0,1))

    os.system('cls')
    mea.shift_orig.max()
    result.max()
    #plt.imshow(mea.modul[:,:,1],cmap='gray')
    #np.amax(mea.mea)
