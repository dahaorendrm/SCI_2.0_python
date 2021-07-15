import numpy as np
from scipy import signal
import pickle
import datetime


from func import utils
logger = utils.init_logger(__name__)



def rgb(orig,mask=None):
    from func.colour_system import cs_srgb_diy as ColorS
    '''
    Need 3D spectral input from 400 - 700 nm

    '''
    if orig.shape[2] > 3:
        orig_rgb = np.zeros((*orig.shape[0:2],3))
        for indx in range(orig.shape[0]):
            for indy in range(orig.shape[1]):
                orig_rgb[indx,indy,:] = ColorS.spec_to_rgb(orig[indx,indy,:])
        orig = orig_rgb
    bayer_mask = np.zeros(orig.shape)
    bayer_mask[::2,::2,0] = 1
    bayer_mask[1::2,::2,1] = 1
    bayer_mask[::2,1::2,1] = 1
    bayer_mask[1::2,1::2,2] = 1
    mea = np.sum(orig * bayer_mask, axis=2)
    return orig, bayer_mask, mea



def cacti(orig, mask, SHIFTD=1):
    logger.info('CASSI model got run.')
    logger.debug("Start shape {}".format(orig.shape))
    (nr,nc,nl) = orig.shape
    if type(SHIFTD) == int:
        modul = utils.shifter(np.repeat(mask[:,:,np.newaxis],nl,axis=2), SHIFTD)
        logger.info('Mask shift at dimension {}.'.format(SHIFTD))
    elif SHIFTD == "3D":
        modul = mask[:,:,:nl]
        logger.info('3D mask.')
    # Generate the measurement
    if SHIFTD is 1:
        mea = np.sum(orig * modul[:,:nc,:], axis=2)
    elif SHIFTD is 0:
        mea = np.sum(orig * modul[:nr,:,:], axis=2)
    else:
        raise error('Shift dimension wrong')
    logger.debug("Maximum value of mea is" + repr(np.amax(mea)))
    return orig, modul, mea

def cassi(orig, mask, SHIFTD=None):
    logger.info('CASSI model got run.')
    logger.debug("Start shape {}".format(orig.shape))
    (nr,nc,nl) = orig.shape
    # Error detection
    #if mask.shape[0:2] != (nr,nc):
    #    logger.error("Dimension mismatched!")
    # Shift the modulations
    if type(SHIFTD) == int:
        logger.debug("Orig shape before shift {}".format(orig.shape))
        orig = utils.shifter(orig, SHIFTD)
        logger.debug("Orig shape after shift {}".format(orig.shape))
        modul = utils.shifter(np.repeat(mask[:,:,np.newaxis],nl,axis=2), SHIFTD)
        logger.info('Mask shift at dimension {}.'.format(SHIFTD))
    elif SHIFTD == "3D":
        modul = mask[:,:,:nl]
        logger.info('3D mask.')
    # Generate the measurement
    mea = np.sum(orig * modul, axis=2)
    logger.debug("Maximum value of mea is" + repr(np.amax(mea)))
    return orig, modul, mea


def lesti_4d(orig, mask, led_curve, SHIFTD=None, CUT_BAND = (4,2), RESAMPLE:'Number of resampled bands' = False):
    logger.info('LESTI_4D model got run.')
    # Cut off the front and end bands in CUT_BAND=(FRONT,END)
    if CUT_BAND:
        orig = orig[:,:,CUT_BAND[0]:-CUT_BAND[1],:]
        led_curve = led_curve[CUT_BAND[0]:-CUT_BAND[1],:]
    # Retrive the necessary dimension
    (nr,nc,nl,nf) = orig.shape
    nled = led_curve.shape[1]
    # Error detection
    if mask.shape[0:2] != (nr,nc):
        logger.error("Dimension mismatched!")
        raise
    if mask.shape[2] < nled * nf:
        logger.error("No enough masks!")
        raise
    else:
        mask = mask[:,:,:nled * nf]
    # generate modulations
    mask_ = np.expand_dims(np.reshape(mask[:,:,:nled*nf],(nr,nc,nled,nf)),axis=2) # Shape:nr,nc,1,nled,nf
    modul = mask_ * np.expand_dims(led_curve,axis=2)  # Shape:nr,nc,nl,nled,nf
    modul = np.sum(modul,axis=3) # Shape:nr,nc,nl,nf
    # Step 1: Produce LED projected images
    orig_leds = np.expand_dims(orig,axis=3)            # shape:nr, nc, nl,    1, nf
    led_curve = np.expand_dims(led_curve,axis=2)       # shape:        nl, nled, 1
    orig_leds = np.sum(orig_leds * led_curve, axis=2)  # shape:nr, nc,     nled, nf
    # Step 2: Impose coding and summation
    mea = np.zeros((nr,nc))
    iled = 0
    for indf in range(nf):
        for iled in range(nled):
            mea += orig_leds[:,:,iled,indf] * mask[:,:,iled + indf * nled]
    # Shift the modulations
    if type(SHIFTD) == int:
        shift_orig = utils.shifter(orig, SHIFTD)
        modul = utils.shifter(modul, SHIFTD)
        logger.info('Mask shift at dimension {}.'.format(SHIFTD))
    else:
        shift_orig = orig
    # Generate the measurement
    mea = np.sum(shift_orig * modul, axis=(2,3))
    # Resample
    if RESAMPLE:
        logger.info('Resample all the data to {} bands'.format(RESAMPLE))
        mea = mea/(nl/RESAMPLE)
        orig = np.swapaxes(orig,3,4)
        orig = np.reshape(orig,(nr*nc*nf,nl))
        orig = signal.resample(orig,RESAMPLE,axis=1)
        orig = np.swapaxes(orig,3,4)
        orig = np.reshape(orig,(nr,nc,RESAMPLE))
        modul = np.swapaxes(modul,3,4)
        modul = np.reshape(modul,(nr*nc*nf,nl))
        modul = signal.resample(modul,RESAMPLE,axis=1)
        modul = np.swapaxes(modul,3,4)
        modul = np.reshape(modul,(nr,nc,RESAMPLE))
    return orig, orig_leds, modul, mask, mea, np.squeeze(led_curve)


def lesti_4d_sst(orig, mask, led_curve, CUT_BAND = (4,2)):
    logger.info('LESTI_4D super spectral-temporal model got run.')
    # Cut off the front and end bands in CUT_BAND=(FRONT,END)
    if CUT_BAND:
        orig = orig[:,:,CUT_BAND[0]:-CUT_BAND[1],:]
        led_curve = led_curve[CUT_BAND[0]:-CUT_BAND[1],:]
    # Retrive the necessary dimension
    (nr,nc,nl,nf) = orig.shape
    nled = led_curve.shape[1]
    # Error detection
    if mask.shape[0:2] != (nr,nc):
        print(f'Shape of mask is {mask.shape}')
        print(f'Shape of orig is {orig.shape}')
        logger.error("Dimension mismatched!")
        raise
    if mask.shape[2] < nf:
        logger.error("No enough masks!")
        raise
    if not nf%nled == 0:
        logger.error("Frame number has to be the multiple of nled.")
        raise
    # Generate modulations
    mask = mask[:,:,:nf]
    modul = np.expand_dims(mask,axis=2) * np.repeat(led_curve,nf//nled,1) # shape:nr, nc, nl, nf
    # Step 1: Produce LED projected images
    orig_leds = np.expand_dims(orig,axis=3)            # shape:nr, nc, nl,    1, nf
    led_curve = np.expand_dims(led_curve,axis=2)       # shape:        nl, nled, 1
    orig_leds = np.sum(orig_leds * led_curve, axis=2)  # shape:nr, nc,     nled, nf
    # Step 2: Impose coding and summation
    mea = np.zeros((nr,nc))
    iled = 0
    for indf in range(nf):
        mea += orig_leds[:,:,iled,indf] * mask[:,:,indf]
        iled += 1
        if iled >= nled:
            iled = 0
    return orig, orig_leds, modul, mask, mea, np.squeeze(led_curve)

def chasti_sst(orig, mask):
    logger.info('CHASTI super temporal model got run.')
    if np.amax(orig) != 1:
        logger.warning('orig max size wrong, and it is '+ str(np.amax(orig)))
    (nr,nc,nl,nf) = orig.shape
    mask = mask[:,:,:nf].astype('float32')
    ind_c = 0
    mea_temp = []
    orig_temp =[]
    for ind in range(nf):
        orig_temp.append(orig[:,:,ind_c,ind])
        mea_temp.append(mask[...,ind] * orig[:,:,ind_c,ind])
        ind_c = ind_c+1 if ind_c<nl-1 else 0
    mea = np.sum(np.asarray(mea_temp),0)
    orig_new = np.moveaxis(np.asarray(orig_temp),0,-1)
    return orig, orig_new, mask, mea

def lesti(orig, mask, led_curve, SHIFTD=None, CUT_BAND = (4,2), RESAMPLE = False):
    logger.info('LESTI_3D model got run.')
    # Cut off the front and end bands in CUT_BAND=(FRONT,END)
    if CUT_BAND:
        orig = orig[:,:,CUT_BAND[0]:-CUT_BAND[1]]
        led_curve = led_curve[CUT_BAND[0]:-CUT_BAND[1],:]
    if orig.ndim is 4:
        orig = orig[...,10]
    # Retrive the necessary dimension
    (nr,nc,nl) = orig.shape
    nled = led_curve.shape[1]
    # Error detection
    if mask.shape[0:2] != (nr,nc):
        logger.error("Dimension mismatched!")
    # Impose LED modulation on the input data

    mask = np.expand_dims(mask[:,:,:nled],axis=2) # Shape:nr,nc,1,nled
    modul = mask * led_curve  # Shape:nr,nc,nl,nled
    modul = np.sum(modul,axis=3) # Shape:nr,nc,nl
    # Shift the modulations
    if type(SHIFTD) == int:
        shift_orig = utils.shifter(orig, SHIFTD)
        modul = utils.shifter(modul, SHIFTD)
        logger.info('Mask shift at dimension {}.'.format(SHIFTD))
    else:
        shift_orig = orig
    # Generate the measurement
    mea = np.sum(shift_orig * modul, axis=2)
    # Resample
    if RESAMPLE:
        logger.info('Resample all the data to {} bands'.format(RESAMPLE))
        mea = mea/(nl/RESAMPLE)
        orig = np.reshape(orig,(nr*nc,nl))
        orig = signal.resample(orig,RESAMPLE,axis=1)
        orig = np.reshape(orig,(nr,nc,RESAMPLE))
        modul = np.reshape(modul,(nr*nc,nl))
        modul = signal.resample(modul,RESAMPLE,axis=1)
        modul = np.reshape(modul,(nr,nc,RESAMPLE))
    return orig, modul, mea

def afunc(orig, modul, SHIFTD=None, SHIFTSTEP=1):
    if type(SHIFTD) == int:
        orig = utils.shifter(orig, SHIFTD, step=SHIFTSTEP)
    dimtuple = tuple(range(2,orig.ndim))
    return np.sum(orig * modul, axis=dimtuple)

def atfunc(mea, modul, SHIFTD=None, SHIFTSTEP=1):
    for ind in range(2,modul.ndim):
        mea = np.expand_dims(mea,ind)
    #dimtuple = tuple(range(2,modul.ndim))
    #mea = np.expand_dims(mea,dimtupl)
    result = mea * modul
    if type(SHIFTD) == int:
        result = utils.shifter(result, SHIFTD, step=SHIFTSTEP, reverse = True)
    return result
