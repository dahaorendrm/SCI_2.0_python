import math
import torch
import numpy as np
from .. import utils
from bm3d import bm3d_deblurring, BM3DProfile, gaussian_kernel
from .hsi.hsicnn import HSI_SDeCNN as hsinet
from .ffdnet.models import FFDNet as ffdnet
from .fastdvdnet.models import FastDVDnet as fastdvdnet
from .fastdvdnet.fastdvdnet import fastdvdnet_seqdenoise
from .spvicnn import Resblock
from skimage.restoration import denoise_tv_chambolle


logger = utils.init_logger(__name__)

def tv_denoiser(x, _lambda, n_iter):
    dt = 0.25
    N = x.shape
    idx = np.arange(1,N[0]+1)
    idx[-1] = N[0]-1
    iux = np.arange(-1,N[0]-1)
    iux[0] = 0
    ir = np.arange(1,N[1]+1)
    ir[-1] = N[1]-1
    il = np.arange(-1,N[1]-1)
    il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)

    for i in range(n_iter):
        z = divp - x*_lambda
        z1 = z[:,ir,:] - z
        z2 = z[idx,:,:] - z
        denom_2d = 1 + dt*np.sqrt(np.sum(z1**2 + z2**2, 2))
        denom_3d = np.tile(denom_2d[:,:,np.newaxis], (1,1,N[2]))
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        divp = p1-p1[:,il,:] + p2 - p2[iux,:,:]
    u = x - divp/_lambda;
    return u

def bm3d_denoiser(x, sigma:'0-1'):
    v = np.zeros((15, 15))
    for x1 in range(-7, 8, 1):
        for x2 in range(-7, 8, 1):
            v[x1 + 7, x2 + 7] = 1 / (x1 ** 2 + x2 ** 2 + 1)
    v = v / np.sum(v)
    for i in range(x.shape[2]):
        x[:,:,i]= bm3d_deblurring(np.atleast_3d(x[:,:,i]), sigma, v)

def vnlnet_denoiser(x):
    theta = vnlnet(np.expand_dims((x1).transpose(2,0,1),3), nsig)
    theta = np.transpose(theta.squeeze(3),(1,2,0))

def fastdvdnet_denoiser_config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device %s is used for denoiser' % (repr(device)))
    model = fastdvdnet(num_input_frames=5, num_color_channels=1)
    model.load_state_dict(torch.load('func/denoiser/fastdvdnet/model_gray.pth',map_location=device))
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    return model,device

def fastdvdnet_denoiser(vnoisy,model,device,sigma):
    """Denoise an input video (H x W x F x C for color video, and H x W x F for
        grayscale video) with FastDVDnet
    """
	# start_time = time.time()
    # nColor = 1 # number of color channels (3 - RGB color, 1 - grayscale)
    NUM_IN_FR_EXT = 5
    # Sets the model in evaluation mode (e.g. it removes BN)
    with torch.no_grad():
        # vnoisy = vnoisy.transpose((2,3,0,1)) # [do it in torch] from H x W x F x C to F x C x H x W
        vnoisy = torch.from_numpy(vnoisy).type('torch.FloatTensor').to(device)
        model = model.type('torch.FloatTensor').to(device)
        noisestd = torch.FloatTensor([sigma]).to(device)
        outv = torch.empty_like(vnoisy)
        #vnoisy = vnoisy.unsqueeze(vnoisy.ndim) # add extra dimension as channel; unsqueeze the color dimension - [H,W,F] to [H,W,F,C=1] or [H,W,S,F] to [H,W,S,F,C=1]
        if vnoisy.ndim is 3:
            vnoisy = vnoisy.unsqueeze(3)
            vnoisy = vnoisy.permute(2, 3, 0, 1)
            outv = fastdvdnet_seqdenoise( seq=vnoisy,
                                          noise_std=noisestd,
                                          windsize=NUM_IN_FR_EXT,
                                          model=model )
            outv = outv.permute(2, 3, 0, 1) # back from F x C x H x W to H x W x F x C
            outv = outv.squeeze(3)
        if vnoisy.ndim is 4:
            for indv in range(vnoisy.size(2)):
                temp = vnoisy[:,:,indv,:]
                temp = temp.unsqueeze(3)
                temp = temp.permute(2, 3, 0, 1) # from H x W x F x C to F x C x H x W
        # print(vnoisy.finfo, noisestd.finfo)
        # print(torch.max(vnoisy),torch.min(vnoisy))
        # vnoisy = torch.clamp(vnoisy,0.,1.)
                temp = fastdvdnet_seqdenoise( seq=temp,
                							  noise_std=noisestd,
                							  windsize=NUM_IN_FR_EXT,
                							  model=model )
        # print(outv.shape)
        # print(torch.max(outv),torch.min(outv))
                temp = temp.permute(2, 3, 1, 0) # back from F x C x H x W to H x W x C x F
                temp = temp.squeeze()
                outv[:,:,indv,:] = temp
        # outv = outv.squeeze(3) # squeeze the color dimension - [H,W,F,C=1] to [H,W,F]
    outv = outv.data.cpu().numpy()
        # outv = outv.transpose((2,3,0,1)) # [do it in torch] back from F x C x H x W to H x W x F x C
	# stop_time = time.time()
	# print('    FastDVDnet video denoising eclipsed in {:.3f}s.'.format(stop_time-start_time))
    return outv

def ffdnet_denoiser_config():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info('Device %s is used for denoiser' % (repr(device)))
    model = ffdnet(num_input_channels=1)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('func/denoiser/ffdnet/net_gray.pth'))
    else:
        state_dict = torch.load('func/denoiser/ffdnet/net_gray.pth',map_location=device)
        state_dict = remove_dataparallel_wrapper(state_dict)
        model.load_state_dict(state_dict)
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    return model,device

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary
	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict

def ffdnet_denoiser(xx,model,device,sigma:'0-1'):
    vshape = xx.shape
    xx = xx.reshape(*vshape[0:2],-1)
    nmask = xx.shape[-1]
    outv = np.zeros(xx.shape)
    for imask in range(nmask):
        # imnoisy = vnoisy[:,:,imask]*255 # to match the scale of the input [0,255]
        imnoisy = xx[:,:,imask] # to match the scale of the input [0,255]
        # from HxWxC to  CxHxW grayscale image (C=1)
        imnoisy = np.expand_dims(imnoisy, 0)
        imnoisy = np.expand_dims(imnoisy, 0)
        imnoisy = torch.Tensor(imnoisy)
        imnoisy = imnoisy.to(device)
        sigma = torch.FloatTensor([sigma])
        sigma = sigma.to(device)
        # Test mode
        with torch.no_grad(): # PyTorch v0.4.0
            imnoisy = torch.autograd.Variable(imnoisy)
            sigma = torch.autograd.Variable(sigma)

        # Estimate noise and subtract it to the input image
        im_noise_estim = model(imnoisy, sigma)
        outim = imnoisy - im_noise_estim
        outv[:,:,imask] = (outim.data.cpu().numpy()[0, 0, :])

    outv = outv.reshape(vshape)
    return outv


    # eimgs = np.zeros(imgs.shape)
    # for i in range(imgs.shape[2]):
    #     im = np.float32(imgs[:,:,i])
    #     im = np.expand_dims(im,2)
    #     im = utils.single2tensor4(im)
    #     im = im.to(device)
    #     sigma = torch.full((1,1,1,1), sigma).type_as(im)
    #     eim = model(im, sigma)
    #     eim = utils.tensor2single(eim)
    #     eimgs[:,:,i] = eim
    # return eimgs

def hsicnn_denoiser_config():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info('Device %s is used for denoiser' % (repr(device)))
    model = hsinet()
    model.load_state_dict(torch.load('func/denoiser/hsi/deep_denoiser.pth'))
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device), device

def hsicnn_denoiser(xx,model,device,it,ch_sigma,it_list, tv_weight=0.5, tv_iter=4):
    #ch_sigma=40/255.
    l_it = []
    for its in it_list:
        if type(its) is int:
            l_it.append(its)
        else:
            l_it.extend([i for i in range(*its)])
    if it in l_it:
        tem = []
        nb = xx.shape[2]
        xx = np.dstack((xx[:, :, 1], xx[:, :, 1], xx[:, :, 1], xx, xx[:, :, -1], xx[:, :, -1], xx[:, :, -1]))
        #logger.debug('newshape of xx is :' + repr(xx.shape))
        for ind in range(nb):
            net_input = xx[:,:,ind:ind+7]
            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
            net_input = net_input.to(device)
            Nsigma = torch.full((1, 1, 1, 1), ch_sigma ).type_as(net_input)
            output = model(net_input, Nsigma)
            output = output.data.squeeze().cpu().numpy()
            tem.append(output)
        return np.dstack(tuple(tem))
    else:
        return denoise_tv_chambolle(xx, tv_weight , n_iter_max=tv_iter, multichannel=True)

def spvicnn_denoiser_config():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('Device %s is used for denoiser' % (repr(device)))
    model = Resblock.__dict__['MultipleBasicBlock2'](input_feature=8, intermediate_feature=128)
    model.load_state_dict(torch.load('func/denoiser/hsi/deep_denoiser.pth'))
    pretrained_weight = torch.load('/lustre/arce/X_MA/SCI_2.0_python/S1_pnp/model-outputs/resnet2/model.pt')
    pretrained_weight = {k[6:]: v for k, v in pretrained_weight.items() }
    model.load_state_dict(pretrained_weight)
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device), device

def spvicnn_denoiser(xx,sigma,it, tv_weight=0.5, tv_iter=7,model=None, device='cpu', it_list=[]):
    l_it = []
    for its in it_list:
        if type(its) is int:
            l_it.append(its)
        else:
            l_it.extend([i for i in range(*its)])
    if it not in l_it:
        return denoise_tv_chambolle(xx, tv_weight , n_iter_max=tv_iter, multichannel=True)

    nb = xx.shape[2]
    if nb%8:
        raise ValueError(f'The image has {nb} channels, which is not fit for spvicnn_denoiser. \
        The number of channels has to be the multiple of 8.')
    #logger.debug('newshape of xx is :' + repr(xx.shape))
    tem = []
    for ind in range(nb//8):
        net_input = xx[:,:,ind*8:ind*8+8]
        net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
        net_input = net_input.to(device)
        output = model(net_input)
        output = output.data.squeeze().permute(1,2,0).cpu().numpy()
        tem.append(output)
    return np.concatenate(tem,2)
