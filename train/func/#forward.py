import json
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import os
from func.Measurement import Measurement

from func import utils
logger = utils.init_logger(__name__)


def forward(JSON_NAME="default_config"):
    with open("config/"+JSON_NAME+".json") as f:
        config = json.load(f)
    logger.info("Read the json file: config/{}.json".format(JSON_NAME))
    logger.debug(config)
    logger.debug("Model name is " + config['MODEL'])

    if config['MODEL'].upper() == "CASSI":
        mea = Measurement(model = config['MODEL'])
        mea.config(config['P_FORWARD'])
        mea.generate(
            orig = load_data('orig',config['P_FORWARD']['ORIG_DATA'],'img'),
            mask = load_data('mask',config['P_FORWARD']['MASK'],'mask'))
    elif config['MODEL'].upper() == "LESTI":
        mea = Measurement(model = config['MODEL'], dim = config['DIM'])
        mea.config(config['P_FORWARD'])
        mea.generate(
            orig = load_data('orig',config['P_FORWARD']['ORIG_DATA'],'img'),
            mask = load_data('mask',config['P_FORWARD']['MASK'],'mask'),
            led_curve = load_data('led',config['P_FORWARD']['BandsLed'],'BandsLed'))


    if config['P_FORWARD']['SAVE_PICKLE']:
        mea.save(config['P_FORWARD']['ORIG_DATA'],config['P_FORWARD']['MASK'])

    return mea

#def add_mea(mea, modul, JSON_NAME="default_config"):
def load_data(path:str,name:str,dataname:str):
    return scio.loadmat(f'data/{path}/{name}.mat')[dataname]



if __name__ == "__main__":
    #JSON_NAME = "config_lesti"
    #(_,_,mea2) = forward()

    #imgplot = plt.imshow(mea2,cmap="gray")
    #plt.savefig('temp/meas.png')
    #plt.close(imgplot)
    #print(data.shape)
    #print(modul.shape)
#     JSON_NAME="default_config"
#     with open("config/"+JSON_NAME+".json") as f:
#         config = json.load(f)
#     mea = Measurement(model = config['P_FORWARD']['MODEL'],
#     orig = scio.loadmat(r'data/orig/{}.mat'.format(config['P_FORWARD']['ORIG_DATA']))['img'],  # input data
#     mask = scio.loadmat(r'data/mask/{}.mat'.format(config['P_FORWARD']['MASK']))['mask']  # input mask
# )
#     type(config['P_FORWARD'])
#     mea.config(config['P_FORWARD'])     mea.generate()
#     print(mea)
#     mea.show()
    mea = forward()
    t = utils.shifter(mea.orig,0)
    t.shape
    mea.shape
    mea.show()
    logger.handlers.clear()
    os.system('cls')
