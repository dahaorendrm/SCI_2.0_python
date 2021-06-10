# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time
import Resblock




class CHASTINET(torch.nn.Module):
    def __init__(self):

        # base class initialization
        super(DAIN, self).__init__()

        #*********************************************
        self.ResNet = Resblock.__dict__['MultipleBasicBlock_4'](11,128)

        self._initialize_weights()


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

    def forward(self, input):
        """
        Parameters
        ----------
        input: shape (batch, stack, width, height)
        -----------
        """
        device = torch.cuda.current_device()
        cur_output_rectified = self.rectifyNet(ResNet) + cur_output[:,0,...]
        return cur_output_rectified
