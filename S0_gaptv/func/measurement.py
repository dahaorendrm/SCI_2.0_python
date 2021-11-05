import numpy as np
import cv2
import scipy.io as scio
import matplotlib.pyplot as plt
from func import utils
#import pickle
import os
import dill
from func.forward_model import *


from func import utils
logger = utils.init_logger(__name__)

class Measurement:
    '''Compressive sensing measurement class'''
    #__slots__ = ('mea','modul','orig','mask','configp','led_curve','afunc','atfunc','modelname','led_curve')
    def __init__(self,model="CASSI", dim=3, configs=None, inputs=None):
        self.modelname = model.upper()
        self.modeldim  = dim
        self.configp = {}
        self.configp["MAXV"] = 255
        self.configp["SCALE_DATA"] = 1
        if "CASSI" in self.modelname:
            self.configp["SHIFTD"] = 0
            self.configp["SHIFTSTEP"] = 1
        if "LESTI" in self.modelname:
            self.configp["SHIFTD"]=None
            self.configp["SCALE_DATA"] = .5
            self.configp["CUT_BAND"] = (4,2)
            self.configp["RESAMPLE"] = False
        if self.modeldim == 4:
            self.configp["NUMF"] = 4
        if configs:
            self.config(configs)
        if inputs:
            self.generate(*inputs)

    def __repr__(self):
        return ("Measurement size:" + repr(self.mea.shape) +
                "\nOriginal data size:" + repr(self.orig.shape) +
                "\nModulation size:" + repr(self.modul.shape)   )

    def __getitem__(self, index):
         return self.mea[index]

    def config(self,dic):
        for k,v in self.configp.items():
            if k in dic.keys():
                self.configp[k] = dic[k]

    def generate(self,orig,mask=None,led_curve=None):
        if not self.configp["SCALE_DATA"] == 1:
            orig = self.scale3d(orig,self.configp["SCALE_DATA"])
        if not self.configp["MAXV"] == 1:
            orig = orig/self.configp["MAXV"]
            logger.debug("Maximum value of orig after conversion is" + repr(np.amax(orig)))
        self.orig = orig
        self.mask = mask
        if self.modelname == "CASSI":
            _, self.modul, self.mea = cassi(self.orig, self.mask,
                                            self.configp["SHIFTD"])
            self.afunc = lambda orig, modul : afunc(orig,modul, self.configp["SHIFTD"])
            self.atfunc = lambda mea, modul : atfunc(mea,modul, self.configp["SHIFTD"])
        elif self.modelname == "LESTI_SST":
            self.led_curve = led_curve
            self.orig = self.orig[...,:self.configp["NUMF"]]
            self.orig, self.orig_leds, self.modul, self.mask, self.mea, \
                        self.led_curve = lesti_4d_sst(self.orig, self.mask,
                        self.led_curve, self.configp["CUT_BAND"])
            self.afunc = lambda orig, modul : afunc(orig,modul)
            self.atfunc = lambda mea, modul : atfunc(mea,modul)
        elif self.modelname == "LESTI" and self.modeldim==4:
            self.led_curve = led_curve
            self.orig = self.orig[...,:self.configp["NUMF"]]
            self.orig, self.orig_leds, self.modul, self.mask, self.mea, \
                             self.led_curve = lesti_4d(self.orig, self.mask, \
                                            self.led_curve,
                                            self.configp["SHIFTD"],
                                            self.configp["CUT_BAND"],
                                            self.configp["RESAMPLE"])
            self.afunc = lambda orig, modul : afunc(orig,modul, self.configp["SHIFTD"])
            self.atfunc = lambda mea, modul : atfunc(mea,modul, self.configp["SHIFTD"])
            #logger.debug('Output the shape of orig, modul, mea: ' + repr(self.orig.shape) + repr(self.modul.shape) + repr(self.mea.shape))
        elif self.modelname == "LESTI" and self.modeldim==3:
            self.led_curve = led_curve
            self.orig, self.orig_leds, self.modul, self.mask, self.mea, \
                       self.led_curve = lesti(self.orig, self.mask, self.led_curve,
                                            self.configp["SHIFTD"],
                                            self.configp["CUT_BAND"],
                                            self.configp["RESAMPLE"])
            self.afunc = lambda orig, modul : afunc(orig,modul, self.configp["SHIFTD"])
            self.atfunc = lambda mea, modul : atfunc(mea,modul, self.configp["SHIFTD"])
        elif self.modelname == "RGB":
            self.orig, self.modul, self.mea = rgb(self.orig, self.mask)
            self.afunc = lambda orig, modul : afunc(orig,modul)
            self.atfunc = lambda mea, modul : atfunc(mea,modul)
        elif self.modelname == "CHASTI_SST":
            _, self.orig, self.modul, self.mea = chasti_sst(self.orig, self.mask)
            self.afunc = lambda orig, modul : afunc(orig,modul)
            self.atfunc = lambda mea, modul : atfunc(mea,modul)
        else:
            raise Error("Can't recognize model type.")
        self.shape = self.mea.shape

    def get_modulation(self):
        return self.modul

    def get_original(self):
        return self.orig

    def scale3d(self,data,scale):
        newdata = [cv2.resize(data[:,:,ind], None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
                    for ind in range(data.shape[2])]
        newdata = np.moveaxis(np.array(newdata),0,-1)
        logger.debug("Resized shape {}".format(newdata.shape))
        return newdata

    def save(self,origname,maskname='None'):
        NAME = "Model={0}_Input={1}_Mask={2}_Shift={3}".format(
                self.modelname,
                origname,
                maskname,
                self.configp['SHIFTD'])
        with open("dataset/"+NAME+".dill",'wb') as f:
            dill.dump(self,f)
        f.close()
        logger.info("Dataset is saved at: " + "dataset/" + NAME +".dill")

    def show(self):
        return plt.imshow(self.mea,cmap="gray")

    @classmethod
    def import_exp_mea_modul(cls, modelname, mea, mask, configs=None):
        '''This function is used to load experiment data to Measurement class.
        Currently only support Juan lab data. More comming soon.'''
        mea_obj = cls(modelname, configs=configs, dim=3)
        mea_obj.modul = mask
        mea_obj.mea = mea
        mea_obj.orig = None
        mea_obj.afunc = lambda orig, modul : afunc(orig,modul, mea_obj.configp["SHIFTD"])
        mea_obj.atfunc = lambda mea, modul : atfunc(mea,modul, mea_obj.configp["SHIFTD"])
        return mea_obj

    @classmethod
    def import_exp_mea_mask(cls, modelname, mea, mask, configs=None):
        '''This function is used to load experiment data to Measurement class.
        Currently only support Juan lab data. More comming soon.'''
        mea_obj = cls(modelname, configs=configs, dim=3)
        mea_obj.modul = utils.shifter(
            np.repeat(mask[:,:,np.newaxis],(mea.shape[1]-mea.shape[0]+1)//mea_obj.configp["SHIFTSTEP"],axis=2),
            mea_obj.configp["SHIFTD"], mea_obj.configp["SHIFTSTEP"])
        mea_obj.mea = mea
        mea_obj.orig = None
        mea_obj.afunc = lambda orig, modul : afunc(orig,modul, mea_obj.configp["SHIFTD"], mea_obj.configp["SHIFTSTEP"])
        mea_obj.atfunc = lambda mea, modul : atfunc(mea,modul, mea_obj.configp["SHIFTD"], mea_obj.configp["SHIFTSTEP"])
        return mea_obj
