import networks.PWCNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from networks import dain as dain_net
import utils
#from torchsummary import summary

logger = utils.init_logger(__name__)

class Motion:
    def __init__(self,method='pwcnet',device=None,timestep=0.5):
        logger.info(('Motion estimation with method: ' + repr(method)))
        self.method = method
        if method is 'dain':
            self.model = dain_slowmotion() # Under construction
        elif method is 'dain_flow':
            self.model = dain_net.__dict__['DAIN_flow'](
                                               timestep=timestep,training=False)
            SAVED_MODEL = './networks/dain/model_weights/best.pth'
            pretrained_dict = torch.load(SAVED_MODEL)
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                                             if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
            # 4. release the pretrained dict for saving memory
            pretrained_dict = []
        elif method is 'dain_flow2':
            self.model = dain_net.__dict__['DAIN_flow2'](
                                               training=False)
            SAVED_MODEL = './networks/dain/model_weights/best.pth'
            pretrained_dict = torch.load(SAVED_MODEL)
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                                             if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
            # 4. release the pretrained dict for saving memory
            pretrained_dict = []
        elif method is 'pwcnet':
            self.model = Run_pwcnet(timestep=timestep)
        self.device = device
        if self.device is None:
            self.device = torch.device(torch.cuda.current_device()
                                        if torch.cuda.is_available() else 'cpu')
        # self.model.to(self.device)
        self.model.cuda()
        self.model.eval()

    def get_motions(self,*args):
        if self.method is 'pwcnet':
            return self.get_pwcnetmotions(*args)
        if self.method is 'dain_flow':
            return self.get_dainflowmotions(*args)
        if self.method is 'dain_flow2':
            return self.get_dainflowmotions2(*args)

    def get_pwcnetmotions(self,startfs,endfs,origs=None):
        self.motion = []
        with torch.no_grad():
            for ind in range(startfs.shape[2]):
                sf = onech2threech(startfs[:,:,ind])
                ef = onech2threech(endfs[:,:,ind])
                # sf = self.pader(sf)
                # ef = self.pader(ef)
                sf = np.expand_dims(sf,0)
                ef = np.expand_dims(ef,0)
                sf = torch.from_numpy(sf).to(self.device)
                ef = torch.from_numpy(ef).to(self.device)
                self.motion.append(self.model(sf,ef))
                logger.info('Frame {}...' % (ind))
        return self.motion

    def get_dainflowmotions2(self,input,origs=None):
        with torch.no_grad():
            input = torch.from_numpy(input).to(self.device)
            output = self.model(input)
        output = output.cpu().numpy()
        # with open("S2_result/dainflow2_results.pickle",'wb') as f:
        #     pickle.dump(output,f)
        # with open("S2_result/dainflow2_results_ref.pickle",'wb') as f:
        #     pickle.dump(origs,f)
        if origs is not None:
            self.psnr = []
            self.ssim = []
            print('output_images shape'+repr(output.shape))
            print('orig shape'+ repr(origs.shape))
            for indr in range(output.shape[3]):
                for indc in range(output.shape[2]):
                    psnr_ = utils.calculate_psnr(
                               origs[:,:,indc,indr],output[:,:,indc,indr])
                    ssim_ = utils.calculate_ssim(
                               origs[:,:,indc,indr],output[:,:,indc,indr])
                    self.ssim.append(ssim_)
                    self.psnr.append(psnr_)
                    logger.debug(
          'Row = %s, Col = %s wrapped image psnr is %s.' % (indr,indc,psnr_))
        return output,self.psnr,self.ssim

    def get_dainflowmotions(self,startfs,endfs,origs=None):
        self.infos = []
        self.outputs = []
        for ind in range(startfs.shape[2]):
            logger.info('Frame %s ...' % ind )
            output, info = self.get_dainflowmotion(startfs[:,:,ind],endfs[:,:,ind])
            if info is not None:
                self.infos.append(info)
            else:
                self.infos=None
            self.outputs.append(output)

        with open("temp/middle_results.pickle",'wb') as f:
            pickle.dump(self.infos,f)
        output_images = self.save_resultsasimgs(self.outputs,self.infos)
        if origs is not None:
            self.psnr = []
            self.ssim = []
            print('output_images shape'+repr(output_images.shape))
            print('orig shape'+ repr(origs.shape))
            for indr in range(output_images.shape[2]):
                for indc in range(output_images.shape[3]):
                    psnr_ = utils.calculate_psnr(
                               origs[:,:,indr,indc],output_images[:,:,indr,indc])
                    ssim_ = utils.calculate_ssim(
                               origs[:,:,indr,indc],output_images[:,:,indr,indc])
                    self.ssim.append(ssim_)
                    self.psnr.append(psnr_)
                    logger.debug(
          'Row = %s, Col = %s wrapped image psnr is %s.' % (indr,indc,psnr_))
        return self.psnr,self.ssim

    def get_dainflowmotion(self,X0,X1):
        X0 = torch.from_numpy(onech2threech(X0)).float()
        X1 = torch.from_numpy(onech2threech(X1)).float()

        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = self.pader(X0)
        X1 = self.pader(X1)
        torch.set_grad_enabled(False)
        X0 = X0.to(self.device)
        X1 = X1.to(self.device)
        X0 = torch.stack((X0, X1),dim = 0)
        assert X0.is_cuda
        output, info = self.model(X0)
        output = self.list2cpu(output)
        #summary(self.model)
        for key in info:
            info[key] = self.list2cpu(info[key])
        #assert not of[0][0].is_cuda
        return output,info

    def list2cpu(self,li):
        li_new = []
        for ind,item in enumerate(li):
            if type(item) is list:
                li_new.append(self.list2cpu(item))
            else:
                li_new.append(item.cpu().numpy())
        return li_new

    @staticmethod
    def pader(data):
        intWidth = data.size(2)
        intHeight = data.size(1)
        channel = data.size(0)
        if not channel == 3:
            return data
        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32
        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight,
                                               intPaddingTop, intPaddingBottom])
        return pader(data)

    @staticmethod
    def images2im(images):
        im0 = np.squeeze(images[0].numpy())
        im1 = np.squeeze(images[1].numpy())
        im0 = np.mean(im0,0)
        im1 = np.mean(im1,0)
        im = np.mean((im0,im1),0)
        return im,im0,im1
    @staticmethod
    def flow2rgb(image):
        image = np.squeeze(image)
        shp = image.shape
        hsvs = np.zeros((*shp[2:4],3,shp[0]))
        hsv  = np.zeros((*image.shape[2:4],3))
        for ind,im in enumerate(image):
            mag, ang = cv2.cartToPolar(im[0,:,:], im[1,:,:])
            hsv[:,:,0] = ang*180/np.pi/2
            hsv[:,:,1] = 255
            hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            hsvs[:,:,:,ind] = cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2BGR)/255.
        return hsvs
    def save_resultsasimgs(self,outputs,infos=None):
        # Output plots save
        output_images = np.zeros((256,256,len(outputs),len(outputs[0])))
        for indr in range(len(outputs)):
            for indc in range(len(outputs[0])):
                tt = outputs[indr][indc]
                tt = np.mean(np.squeeze(tt),0)
                output_images[:,:,indr,indc] = tt
        fig = utils.display_highdimdatacube(output_images)
        savename = 'output_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        if infos is None:
            return output_images
        # Plots save
        depth_images = [info['depth'] for info in infos]
        flow_f_ims = [self.flow2rgb(info['flow_f']) for info in infos]
        flow_inv_ims = [self.flow2rgb(info['flow_inv']) for info in infos]
        wflow_f_ims = [self.flow2rgb(info['wrappedflow_f']) for info in infos]
        wflow_inv_ims = [self.flow2rgb(info['wrappedflow_inv']) for info in infos]
        ctx_f_images = [info['context_f'] for info in infos]
        ctx_inv_images = [info['context_inv'] for info in infos]
        kerl_f_images = [info['kernel_f'] for info in infos]
        kerl_inv_images = [info['kernel_inv'] for info in infos]
        mean_images = [
        [np.mean((info['wrapped_ims_f'][ind],info['wrapped_ims_inv'][ind]),0)
                                for ind in range(len(info['wrapped_ims_f']))]
         for info in infos]
        depth_images = np.squeeze(depth_images)
        depth_images = np.asarray(depth_images)
        depth_images = np.moveaxis(depth_images,[0,1],[-1,-2])
        flow_f_ims = np.squeeze(flow_f_ims)
        flow_f_ims = np.asarray(flow_f_ims)
        flow_f_ims = np.moveaxis(flow_f_ims,0,-1)
        flow_inv_ims = np.asarray(flow_inv_ims)
        flow_inv_ims = np.moveaxis(flow_inv_ims,0,-1)
        wflow_f_ims = np.asarray(wflow_f_ims)
        wflow_f_ims = np.moveaxis(wflow_f_ims,0,-1)
        wflow_inv_ims = np.asarray(wflow_inv_ims)
        wflow_inv_ims = np.moveaxis(wflow_inv_ims,0,-1)
        ctx_f_images = np.squeeze(ctx_f_images)
        ctx_f_images = np.asarray(ctx_f_images)
        ctx_f_images = np.moveaxis(ctx_f_images,[0,1],[-1,-2])
        ctx_f_images = ctx_f_images[:,:,0:5,:]
        ctx_inv_images = np.squeeze(ctx_inv_images)
        ctx_inv_images = np.asarray(ctx_inv_images)
        ctx_inv_images = np.moveaxis(ctx_inv_images,[0,1],[-1,-2])
        ctx_inv_images = ctx_inv_images[:,:,0:5,:]
        kerl_f_images = np.squeeze(kerl_f_images)
        kerl_f_images = np.asarray(kerl_f_images)
        kerl_f_images = np.moveaxis(kerl_f_images,[0,1],[-1,-2])
        kerl_inv_images = np.squeeze(kerl_inv_images)
        kerl_inv_images = np.asarray(kerl_inv_images)
        kerl_inv_images = np.moveaxis(kerl_inv_images,[0,1],[-1,-2])
        mean_images = np.squeeze(mean_images)
        mean_images = np.asarray(mean_images)
        mean_images = np.moveaxis(mean_images,[0,1],[-1,-2])

        print(depth_images.shape)
        print(flow_f_ims.shape)
        print(flow_inv_ims.shape)
        print(wflow_f_ims.shape)
        print(wflow_inv_ims.shape)
        print(ctx_f_images.shape)
        print(ctx_inv_images.shape)
        print(kerl_f_images.shape)
        print(kerl_inv_images.shape)
        print(mean_images.shape)
        '''
        fig = utils.display_highdimdatacube(depth_images)
        savename = 'depth_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('Depth images saved')
        fig = utils.display_highdimdatacube(flow_f_ims,rgb=True)
        savename = 'flow_f_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('flow f images saved')
        fig = utils.display_highdimdatacube(flow_inv_ims,rgb=True)
        savename = 'flow_inv_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('flow inv images saved')
        fig = utils.display_highdimdatacube(wflow_f_ims,rgb=True)
        savename = 'wflow_f_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('w flow f images saved')
        fig = utils.display_highdimdatacube(wflow_inv_ims,rgb=True)
        savename = 'wflow_inv_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('wflow inv images saved')
        fig = utils.display_highdimdatacube(ctx_f_images)
        savename = 'ctx_f_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('ctx f images saved')
        fig = utils.display_highdimdatacube(ctx_inv_images)
        savename = 'ctx_inv_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('ctx inv images saved')
        fig = utils.display_highdimdatacube(kerl_f_images)
        savename = 'kerl_f_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('kerl f images saved')
        fig = utils.display_highdimdatacube(kerl_inv_images)
        savename = 'kerl_inv_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('kerl inv images saved')
        fig = utils.display_highdimdatacube(mean_images)
        savename = 'mean_images_results'
        fig.savefig('./temp/%s.png' % (savename), dpi = 1000)
        print('mean images saved')
        '''
        return output_images



def motion2rgb(motion):
    tt = motion.numpy()
    tt = np.squeeze(tt)
    mag, ang = cv2.cartToPolar(tt[0,:,:], tt[1,:,:])
    hsv = np.zeros((*tt.shape[1:3],3))
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2BGR)

def onech2threech(frame):
    return np.repeat(np.expand_dims(frame,0),3,0)


class Run_pwcnet(nn.Module):
    def __init__(self, timestep=0.5):
        super(Run_pwcnet, self).__init__()
        self.timestep = timestep
        self.pwcnet = PWCNet.__dict__['pwc_dc_net']("PWCNet/pwc_net.pth.tar")

    def forward(self,startf,endf):
        numFrames =int(1.0/self.timestep) - 1
        time_offsets = [ self.timestep * ind for ind in range(1, 1+numFrames)]
        cur_offset_input = torch.cat((startf, endf), dim=1)
        return [
            forward_pwcnet(self.pwcnet, cur_offset_input, time_offsets=time_offsets),
            forward_pwcnet(self.pwcnet, torch.cat((cur_offset_input[:, 3:, ...],
                                cur_offset_input[:, 0:3, ...]), dim=1),
                      time_offsets=time_offsets[::-1])
            ]


def forward_pwcnet(model, input, time_offsets = None):

    if time_offsets == None:
        time_offsets = [0.5]
    elif type(time_offsets) == float:
        time_offsets = [time_offsets]
    elif type(time_offsets) == list:
        pass
    temp = model(input.float())  # this is a single direction motion results, but not a bidirectional one
    temp = temp.cpu()
    temps = [20.0 * temp * time_offset for time_offset in time_offsets]# single direction to bidirection should haven it.
    temps = [nn.Upsample(scale_factor=4, mode='bilinear')(temp)
                                                     for temp in temps]# nearest interpolation won't be better i think
    return temps
