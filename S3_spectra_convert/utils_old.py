# logger
import logging
import numpy as np
import math
import cv2
import functools
import time, os
import reprlib
import matplotlib.pyplot as plt

def display_highdimdatacube(data,rgb=False,transpose=False):
    if rgb is True:
        assert data.ndim >= 3 and data.ndim <= 5
        if data.ndim is 4:
            fig = plt.figure(figsize=(16,16),
                       dpi = (data.shape[3]+1)*256/16, constrained_layout=True)
            ax = fig.subplots(data.shape[3])
            for indr in range(data.shape[3]):
                ax[indr].imshow(data[:,:,:,indr])
                ax[indr].axis('off')
        elif data.ndim is 5:
            if transpose:
                data = np.swapaxes(data,3,4)
            fig = plt.figure(figsize=(16,16),
                       dpi = (data.shape[3]+1)*256/16, constrained_layout=True)
            ax = fig.subplots(data.shape[3],data.shape[4],squeeze=False)
            for indr in range(data.shape[3]):
                for indc in range(data.shape[4]):
                    ax[indr][indc].imshow(data[:,:,:,indr,indc])
                    ax[indr][indc].axis('off')
        else:
            fig, ax = plt.subplots(constrained_layout=True)
            ax.imshow(data)
            ax.axis('off')
        return fig
    assert data.ndim >= 2 and data.ndim <= 4
    if data.ndim is 3:
        fig = plt.figure(figsize=(16,16),
                       dpi = (data.shape[2]+1)*256/16, constrained_layout=True)
        ax = fig.subplots(1,data.shape[2])
        for indr in range(data.shape[2]):
            ax[indr].imshow(data[:,:,indr],cmap='gray')
            ax[indr].axis('off')
    elif data.ndim is 4:
        if transpose:
            data = np.swapaxes(data,2,3)
        fig = plt.figure(figsize=(16,16),
                       dpi = (data.shape[3]+1)*256/16, constrained_layout=True)
        ax = fig.subplots(data.shape[2],data.shape[3],squeeze=False)
        for indr in range(data.shape[2]):
            for indc in range(data.shape[3]):
                ax[indr][indc].imshow(data[:,:,indr,indc],cmap='gray')
                ax[indr][indc].axis('off')
    else:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.imshow(data,cmap='gray')
        ax.axis('off')
    return fig


def timer(logger=None):
    def decorate(func):
        @functools.wraps(func)
        def timer_(*args, **kwargs):
            t0 = time.time()
            results = func(*args, **kwargs)
            elapsed = time.time() - t0
            name = func.__name__
            arg_lst = []
            if args:
                for arg in args:
                    if hasattr(arg, 'shape'):
                        arg_lst.append('%s with shape %s' %
                                        (reprlib.repr(arg), repr(arg.shape)))
                    else:
                        arg_lst.append(repr(arg))
            if kwargs:
                pairs = []
                for k,w in kwargs.items():
                    if hasattr(w, 'shape'):
                        pairs.append('%s=%s with shape %s' %
                                         (k,reprlib.repr(w),repr(w.shape)))
                    else:
                        pairs.append('%s=%s' % (k,w))
                arg_lst.append(', \n'.join(pairs))
            arg_str = ', \n'.join(arg_lst)
            res_lst = []
            if type(results) is tuple:
                for result in results:
                    res_lst.append('%s' % (reprlib.repr(result)))
            else:
                res_lst.append('%s' % (reprlib.repr(results)))
            res_str = ', '.join(res_lst)
            logger.info('[%0.4fs] %s(%s) \n-> %s' %
                                       (elapsed, name, arg_str, res_str))
            return results
        return timer_
    return decorate



def init_logger(NAME):
    logger = logging.getLogger(NAME)
    logger.setLevel(logging.DEBUG)
    try:
        os.mkdir('temp')
    except:
        pass
    open('temp/test.log', "w")
    fh = logging.FileHandler('temp/test.log')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
                        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("Logger {} initiated.".format(NAME))
    return logger


'''
# --------------------------------------------
# metric, PSNR and SSIM
# --------------------------------------------
'''

# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, maxv=1, border=0):
    # img1 and img2 have range [0, 1]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    # img1 = img1/np.amax(img1)
    # img1 = img1/np.amax(img1)
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(maxv / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, max_v=1, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1 = img1/max_v*255
    img2 = img2/max_v*255
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        ssims = []
        for i in range(img1.shape[2]):
            ssims.append(ssim(np.squeeze(img1[...,i]), np.squeeze(img2[...,i])))
        return np.array(ssims).mean()
    elif img1.ndim == 4:
        ssims = []
        for j in range(img1.shape[3]):
            for i in range(img1.shape[2]):
                ssims.append(ssim(np.squeeze(img1[...,i,j]),
                                                  np.squeeze(img2[...,i,j])))
        return np.array(ssims).mean()

    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                     ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
    result =  ssim_map.mean()
    if result is None:
        return 0
    return result

def shifter(data, shiftd, step=1, reverse=False):
    '''
    data   : Input 3d datacube
    shiftd : Which dimension to shift
    step   : Shifing steps
    reverse: Reverse shifting
    '''
    if not (isinstance(shiftd,int) or isinstance(shiftd,float)):
        return data
    if shiftd != 0:
        data = np.swapaxes(data,0,shiftd)
    datashape = list(data.shape)
    if reverse:
        datashape[0] = datashape[0] - datashape[2]*step
        newdata = np.zeros(datashape)
        for i in range(datashape[2]):
            sl = slice(i*step,i*step+datashape[0])
            newdata[:,:,i] = data[sl,:,i]
    else:
        datashape[0] = datashape[0] + datashape[2]*step
        newdata = np.zeros(datashape)
        for i in range(datashape[2]):
            sl = slice(i*step,i*step+datashape[0] - datashape[2]*step)
            newdata[sl,:,i] = data[:,:,i]
    if shiftd != 0:
        data = np.swapaxes(data,0,shiftd)
        newdata = np.swapaxes(newdata,0,shiftd)
    return newdata

def load_data(path:str,name:str,dataname:str):
    return scio.loadmat(f'data/{path}/{name}.mat')[dataname]

# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).\
                                permute(2, 0, 1).float().unsqueeze(0)


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

# convert torch tensor to single
def tensor2single3(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).\
                             permute(2, 0, 1, 3).float().unsqueeze(0)


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).\
                                    float().unsqueeze(0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).\
                                    permute(2, 0, 1, 3).float()
