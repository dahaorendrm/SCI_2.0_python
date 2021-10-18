# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .my_package.FilterInterpolation import  FilterInterpolationModule
from .my_package.FlowProjection import  FlowProjectionModule #,FlowFillholeModule
from .my_package.DepthFlowProjection import DepthFlowProjectionModule

from .Stack import Stack

from . import PWCNet
from . import S2D_models
from . import Resblock
from . import MegaDepth
import time
import utils
logger = utils.init_logger(__name__)
class DAIN_flow2(torch.nn.Module):
    def __init__(self,
                 filter_size = 4,
                 training=True):

        # base class initialization
        super(DAIN_flow2, self).__init__()
        self.device = torch.cuda.current_device()
        self.filter_size = filter_size
        self.training = training
        i = 0
        channel = 3
        self.initScaleNets_filter,self.initScaleNets_filter1,self.initScaleNets_filter2 = \
            self.get_MonoNet5(channel if i == 0 else channel + filter_size * filter_size, filter_size * filter_size, "filter")

        self.ctxNet = S2D_models.__dict__['S2DF_3dense']()
        self.ctx_ch = 3 * 64 + 3

        self.rectifyNet = Resblock.__dict__['MultipleBasicBlock_4'](3 + 3 + 3 +2*1+ 2*2 +16*2+ 2 * self.ctx_ch,128)

        self._initialize_weights()

        if self.training:
            self.flownets = PWCNet.__dict__['pwc_dc_net']("PWCNet/pwc_net.pth.tar")
        else:
            self.flownets = PWCNet.__dict__['pwc_dc_net']()
        self.div_flow = 20.0

        #extract depth information
        if self.training:
            self.depthNet=MegaDepth.__dict__['HourGlass']("dain/MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")
        else:
            self.depthNet=MegaDepth.__dict__['HourGlass']()

        return

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(m)
                count+=1
                # print(count)
                # weight_init.xavier_uniform(m.weight.data)
                nn.init.xavier_uniform_(m.weight.data)
                # weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:
            #     print(m)
    @staticmethod
    def onech2threech(frame):
        if torch.is_tensor(frame):
            return frame.repeat(3,1,1)
        else:
            return np.repeat(np.expand_dims(frame,0),3,0)

    def forward(self, input):

        """
        Parameters
        ----------
        input: shape (ngroup, nframe, width, height)
        -----------
        """
        assert input.is_cuda
        input = input.type(torch.cuda.FloatTensor)
        ngroup, nf, width, height = input.size()
        # s1 = torch.cuda.Stream(device=device, priority=5)
        # s2 = torch.cuda.Stream(device=device, priority=10) #PWC-Net is slow, need to have higher priority

        self.training = False

        result = torch.empty(width,height,nf,ngroup*nf).type(torch.cuda.FloatTensor)
        for indg in range(ngroup-1):
            '''
            Step 1: Fill 8(nf)*7 intermediate frames
            '''
            for indf in range(nf):
                ### Step 1.1: Load two input frames and put them in results
                input0 = self.onech2threech(input[indg,  indf,...])
                input1 = self.onech2threech(input[indg+1,indf,...])
                result[:,:,indf, nf* indg   +indf] = torch.mean(input0,0)
                if indg == ngroup-2:
                    result[:,:,indf, nf*(indg+1)+indf] = torch.mean(input1,0)
                ### Step 1.2: Fill the 7 intermediate frames and put them in results
                li_7 = self.forward_7frames(input0,input1,nf)
                print('Length of li_7 is '+str(len(li_7))+' with shape '+str(li_7[0].size()))
                for ind,item in enumerate(li_7):
                    item = torch.squeeze(item)
                    result[:,:, indf, nf * indg + indf + ind + 1] = torch.mean(item,0)
                    print('li7 output index col='+str(indf)+' ,row='+str(nf * indg + indf + ind + 1))
            '''
            Step 2: Generate images for the edge groups
                Flow generated from input0->input1
                Apply the generated flow on input2
            '''
            if indg == 0:
                for indf in range(nf-1):
                    input1 = self.onech2threech(result[:,:,indf,indf])
                    for indf2 in range(indf+1,nf):
                        input0 = self.onech2threech(result[:,:,indf,indf2])
                        input2 = self.onech2threech(result[:,:,indf2,indf2])
                        input0 = self.forward_simplewrap(input0,input1,input2,rectify=True)
                        input0 = torch.squeeze(input0)
                        result[:,:,indf2,indf] = torch.mean(input0,0)
                        print(f'Generate flow from img({indf},{indf2}) to img({indf},{indf}), apply on img({indf2},{indf2})')
                        print('pre output index col='+str(indf2)+' ,row='+str(indf))

            if indg == ngroup-2:
                for indf in reversed(range(1,nf)):
                    input1 = self.onech2threech(result[:,:,indf,nf*(indg+1)+indf])
                    for indf2 in range(indf):
                        input0 = self.onech2threech(result[:,:,indf,nf*(indg+1)+indf2])
                        input2 = self.onech2threech(result[:,:,indf2,nf*(indg+1)+indf2])
                        input0 = self.forward_simplewrap(input0,input1,input2,rectify=True)
                        input0 = torch.squeeze(input0)
                        result[:,:,indf2,nf*(indg+1)+indf] = torch.mean(input0,0)
                        print(f'Generate flow from img({indf},{nf*(indg+1)+indf2}) to img({indf},{nf*(indg+1)+indf}), apply on img({indf2},{indf2,nf*(indg+1)+indf2})')
                        print('post output index col='+str(indf2)+' ,row='+str(nf*(indg+1)+indf))
        #result.cpu()
        return result

    def forward_7frames(self,input0,input1,nf):
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()

        cur_input_0 = torch.unsqueeze(input0,0)
        cur_input_1 = torch.unsqueeze(input1,0)
        '''
            STEP 1: concatenating the inputs.
        '''
        cur_offset_input = torch.cat((cur_input_0, cur_input_1), dim=1)
        cur_filter_input = cur_offset_input
        #numFrames =int(1.0/0.125) - 1
        numFrames = nf-1
        unit_time = 1.0/nf
        time_offsets = [ kk * unit_time for kk in range(1, 1+numFrames,1)]
        '''
            STEP 2: First layer estimations
        '''
        #print('Shape of cur_offset_input'+str(cur_offset_input.size()))
        #print('Shape of cur_filter_input'+str(cur_filter_input.size()))
        with torch.cuda.stream(s1):
            '''
                STEP 2.1: Depth estimation
            '''
            temp  = self.depthNet(
                    torch.cat((cur_filter_input[:, :3, ...],
                               cur_filter_input[:, 3:, ...]),dim=0))
            log_depth = [temp[:cur_filter_input.size(0)],
                                temp[cur_filter_input.size(0):]]
            depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]
            '''
                STEP 2.2: context estimation
                Depth information is also included in the context estimation
            '''
            cur_ctx_output = [
                torch.cat((self.ctxNet(cur_filter_input[:, :3, ...]),
                       log_depth[0].detach()), dim=1),
                    torch.cat((self.ctxNet(cur_filter_input[:, 3:, ...]),
                   log_depth[1].detach()), dim=1)
                    ]
            '''
                STEP 2.3: kernel estimation
            '''
            temp = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input, 'filter')
            cur_filter_output = [self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                             self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)]
        with torch.cuda.stream(s2):
            '''
                STEP 2.4: Flow estimation
            '''
            cur_offset_outputs = [
                    self.forward_flownets(
                    self.flownets, cur_offset_input, time_offsets=time_offsets),
                    self.forward_flownets(
                    self.flownets, torch.cat((cur_offset_input[:, 3:, ...],
                                        cur_offset_input[:, 0:3, ...]), dim=1),
                                               time_offsets=time_offsets[::-1])]
            torch.cuda.synchronize() #synchronize s1 and s2
        '''
            STEP 3: Flow projections
        '''
        cur_offset_outputs = [
           self.FlowProject(cur_offset_outputs[0],depth_inv[0]),
           self.FlowProject(cur_offset_outputs[1],depth_inv[1])]
        '''
            STEP 4: Estimate wrapped frames
        '''
        cur_output_rectified = []
        cur_output = []
        for temp_0,temp_1, timeoffset in zip(
                    cur_offset_outputs[0], cur_offset_outputs[1], time_offsets):
            cur_offset_output = [temp_0,temp_1]
            '''
                STEP 4.1: Wrap frames with flow and kernel
            '''
            weighted_sum ,ref0,ref1 = \
              self.FilterInterpolate(cur_input_0, cur_input_1,cur_offset_output,
                              cur_filter_output,self.filter_size**2, timeoffset)
            cur_output.append(weighted_sum)
            '''
                STEP 4.2: Wrap context with flow and kernel
            '''
            ctx0,ctx1 = self.FilterInterpolate_ctx(
                    cur_ctx_output[0],cur_ctx_output[1],
                               cur_offset_output,cur_filter_output, timeoffset)
            '''
                STEP 4.3: Concatenation
                Interpolated frames; interpolated flow; kernel filters; context
            '''
            rectify_input = torch.cat((weighted_sum,ref0,ref1,
                                    cur_offset_output[0],cur_offset_output[1],
                                    cur_filter_output[0],cur_filter_output[1],
                                    ctx0,ctx1),dim =1)
            '''
                STEP 4.3: Rectification with resnet
            '''
            #print('Shape of 7 rectify input'+str(rectify_input.size()))
            rectified_temp = self.rectifyNet(rectify_input) + weighted_sum
            #rectified_temp = nn.Sigmoid()(rectified_temp)
            cur_output_rectified.append(rectified_temp)
        return cur_output_rectified

    def forward_simplewrap(self,input0,input1,input2,rectify=True):
        '''
        Generate flow from input0 to input1, then apply the flow to input2
        '''
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()

        cur_input_0 = torch.unsqueeze(input0,0)
        cur_input_1 = torch.unsqueeze(input1,0)
        cur_input_2 = torch.unsqueeze(input2,0)
        '''
            STEP 1: concatenating the inputs.
        '''
        cur_offset_input = torch.cat((cur_input_1,cur_input_0), dim=1) # shape:1,6,256,256
        cur_filter_input = cur_input_2 # shape:1,3,256,256
        '''
            STEP 2: First layer estimations
        '''
        with torch.cuda.stream(s1):
            '''
                STEP 2.1: Depth estimation
            '''
            log_depth  = self.depthNet(cur_filter_input)
            depth_inv = 1e-6 + 1 / torch.exp(log_depth)
            '''
                STEP 2.2: context estimation
                Depth information is also included in the context estimation
            '''
            cur_ctx_output = torch.cat((self.ctxNet(cur_filter_input),
                                               log_depth.detach()), dim=1)

        with torch.cuda.stream(s2):
            '''
                STEP 2.3: Kernel estimation
                ######### maybe should from input2?
            '''
            temp = self.forward_singlePath(self.initScaleNets_filter, torch.cat((cur_input_2,cur_input_2), dim=1), 'filter')
            cur_filter_output = [self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                             self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)]
            '''
                STEP 2.4: Flow estimation
            '''
            cur_offset_output = self.forward_flownets(self.flownets, cur_offset_input)
        '''
            STEP 3: Estimate wrapped frames
        '''
        '''
            STEP 3.1: Wrap frames with flow and kernel
            ####### Use input2 kernel filter?
        '''
        #ref0_offset = FilterInterpolationModule()(
        #                        cur_input_2, cur_offset_output,cur_filter_output[0].detach())
        ref0_offset = self.flownets.warp(cur_input_2,cur_offset_output*20)
        #ref0_offset = torch.unsqueeze(ref0_offset,0)
        #return ref0_offset
        '''
            STEP 3.2: Wrap context with flow and kernel
        '''
        ctx0_offset = FilterInterpolationModule()(cur_ctx_output,
                                 cur_offset_output,cur_filter_output[0].detach())
        '''
            STEP 3.3: Concatenation
            Interpolated frames; interpolated flow; kernel filters; context
        '''
        if rectify:
            #print('Shape of input:(1)ref0'+str(ref0_offset.size())+'(2)cur_offset_outputs'+str(cur_offset_output.size())+'(3)cur_filter_output'+str(cur_filter_output[0].size())+'(4)context'+str(ctx0_offset.size()))
            rectify_input = torch.cat((ref0_offset,ref0_offset,ref0_offset,
                                    cur_offset_output,cur_offset_output,
                                    cur_filter_output[0],cur_filter_output[1], # cur_filter_output[0] again?
                                    ctx0_offset,ctx0_offset),dim = 1)
            '''
                STEP 4.3: Rectification with resnet
            '''
            #print('Shape of simple rectify input'+str(rectify_input.size()))
            cur_output_rectified = self.rectifyNet(rectify_input) + ref0_offset
            # cur_output_rectified = nn.Sigmoid()(cur_output_rectified)
            return cur_output_rectified
        else:
            return ref0_offset
            #return cur_offset_output

    def forward_flownets(self, model, input, time_offsets = None):
        temp = model(input)  # this is a single direction motion results, but not a bidirectional one
        if time_offsets == None :
            return nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)(temp)
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        temps = [self.div_flow * temp * time_offset for time_offset in time_offsets]# single direction to bidirection should haven it.
        temps = [nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)(temp)  for temp in temps]# nearest interpolation won't be better i think
        return temps

    '''keep this function'''
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:  # self.initScaleNets_offset:
            # print(type(layers).__name__)
            # print(k)
            # if k == 27:
            #     print(k)
            #     pass
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp

    '''keep this funtion'''
    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))

        return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    '''keep this function'''
    @staticmethod
    def FlowProject(inputs, depth = None):
        if depth is not None:
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input,depth) for input in inputs]
        else:
            outputs = [ FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        return outputs


    '''keep this function'''
    @staticmethod
    def FilterInterpolate_ctx(ctx0,ctx2,offset,filter, timeoffset):
        ##TODO: which way should I choose

        ctx0_offset = FilterInterpolationModule()(ctx0,offset[0].detach(),filter[0].detach())
        ctx2_offset = FilterInterpolationModule()(ctx2,offset[1].detach(),filter[1].detach())

        return ctx0_offset, ctx2_offset
        # ctx0_offset = FilterInterpolationModule()(ctx0.detach(), offset[0], filter[0])
        # ctx2_offset = FilterInterpolationModule()(ctx2.detach(), offset[1], filter[1])
        #
        # return ctx0_offset, ctx2_offset
    '''Keep this function'''
    @staticmethod
    def FilterInterpolate(ref0, ref2, offset, filter,filter_size2, time_offset):
        ref0_offset = FilterInterpolationModule()(ref0, offset[0],filter[0])
        ref2_offset = FilterInterpolationModule()(ref2, offset[1],filter[1])

        # occlusion0, occlusion2 = torch.split(occlusion, 1, dim=1)
        # print((occlusion0[0,0,1,1] + occlusion2[0,0,1,1]))
        # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset) / (occlusion0 + occlusion2)
        # output = * ref0_offset + occlusion[1] * ref2_offset
        # automatically broadcasting the occlusion to the three channels of and image.
        # return output
        # return ref0_offset/2.0 + ref2_offset/2.0, ref0_offset,ref2_offset
        return ref0_offset*(1.0 - time_offset) + ref2_offset*(time_offset), ref0_offset, ref2_offset

    '''keep this function'''
    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                        padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
        )
        return layers


    '''keep this fucntion'''
    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                        padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False)
        ])
        return layers

    '''keep this function'''
    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                            padding,kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            # nn.BatchNorm2d(output_filter),

            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers

    '''klkeep this function'''
    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                            padding,unpooling_factor):

        layers = nn.Sequential(*[

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear',align_corners=True),

            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            # nn.BatchNorm2d(output_filter),


            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers
