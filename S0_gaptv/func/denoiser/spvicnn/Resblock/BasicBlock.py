import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as weight_init
import torch
__all__ = ['MultipleBasicBlock','MultipleBasicBlock_4','BasicBlock','MultipleCascadeBlock','MultipleCascadeBlock_func','MultipleBasicBlock2']
def conv3x3(in_planes, out_planes, dilation=1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.downsample = downsample
        self.stride = stride
        #self.do = nn.Dropout2d(p=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #x = self.do(x)
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        #if self.downsample is not None:
        #    residual = self.downsample(x)
        out += residual
        #out = self.LeakyReLU(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU1 = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, round(planes/2))
        self.ReLU2 = nn.LeakyReLU(inplace=True)
        self.conv3 = conv3x3(round(planes/2),inplanes)
        self.bn2 = nn.BatchNorm2d(round(planes/2))
        #self.downsample = downsample
        self.stride = stride
        #self.do = nn.Dropout2d(p=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #x = self.do(x)
        copy = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ReLU2(out)
        out = self.conv3(out)
        #if self.downsample is not None:
        #    residual = self.downsample(x)
        out += copy
        #out = self.LeakyReLU(out)
        return out

class CascadeBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, features, dilation = 1, stride=1, downsample=None):
        super(CascadeBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, features,dilation, stride)
        #self.bn1 = nn.BatchNorm2d(features)
        self.ReLU1 = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(features, int(features/2))
        self.ReLU2 = nn.LeakyReLU(inplace=True)
        self.conv3 = conv3x3(int(features/2), 1)
        #self.bn2 = nn.BatchNorm2d(features)
        #self.downsample = downsample
        self.stride = stride
        #self.do = nn.Dropout2d(p=0.2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.ReLU1(out)
        out = self.conv2(out)
        out = self.ReLU2(out)
        out = self.conv3(out)

        #out = self.bn2(out)
        #if self.downsample is not None:
        #    residual = self.downsample(x)
        return x[:,0:1,:]+out


class MultipleCascadeBlock(nn.Module):
    def __init__(self,intermediate_feature = 64, dense = True):
        super(MultipleCascadeBlock, self).__init__()
        self.dense = dense
        self.intermediate_feature = intermediate_feature

        self.block1 = CascadeBlock(3,intermediate_feature*2)
        self.block2 = CascadeBlock(2,intermediate_feature)
        self.block3 = CascadeBlock(2,intermediate_feature)
        self.block4 = CascadeBlock(2,intermediate_feature)
        self.block5 = CascadeBlock(1,64)

        self.BN1 = nn.BatchNorm2d(3)
        self.BN2 = nn.BatchNorm2d(2)
        self.BN3 = nn.BatchNorm2d(2)
        self.BN4 = nn.BatchNorm2d(2)
        self.BN5 = nn.BatchNorm2d(1)
        # self.do1 = nn.Dropout2d(p=0.2,inplace=True)
        # self.do2 = nn.Dropout2d(p=0.2,inplace=True)
        # self.do3 = nn.Dropout2d(p=0.2,inplace=True)
        # self.do4 = nn.Dropout2d(p=0.2,inplace=True)
        # self.do5 = nn.Dropout2d(p=0.2,inplace=True)
        #self.BN7     = nn.BatchNorm2d(intermediate_feature)
        #self.BNend     = nn.BatchNorm2d(intermediate_feature)
        #self.endlayer = nn.Sequential(*[nn.Conv2d(intermediate_feature, 1 , 5, 1, 2),nn.Sigmoid()])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        step1 = x[:,:3,...]
        # step1 = self.do1(step1)
        step1 = self.BN1(step1)
        step1 = self.block1(step1)

        step2 = torch.cat((step1,x[:,4:5,...]),1)
        # step2 = self.do2(step2)
        step2 = self.BN2(step2)
        step2 = self.block2(step2)

        step3 = torch.cat((step2,x[:,3:4,...]),1)
        # step3 = self.do3(step3)
        step3 = self.BN3(step3)
        step3 = self.block3(step3)

        step4 = torch.cat((step3,x[:,5:6,...]),1)
        # step4 = self.do4(step4)
        step4 = self.BN4(step4)
        step4 = self.block4(step4)

        step5 = step4
        # step5 = self.do5(step5)
        step5 = self.BN5(step5)
        step5 = self.block5(step5)

        return step5

class MultipleBasicBlock2(nn.Module):

    def __init__(self,input_feature,
                 intermediate_feature = 128, dense = True):
        super(MultipleBasicBlock2, self).__init__()
        self.input_feature = input_feature
        self.spatblock1 = BasicBlock2(1, intermediate_feature) if input_feature>=1 else None
        self.spatblock2 = BasicBlock2(1, intermediate_feature) if input_feature>=2 else None
        self.spatblock3 = BasicBlock2(1, intermediate_feature) if input_feature>=3 else None
        self.spatblock4 = BasicBlock2(1, intermediate_feature) if input_feature>=4 else None
        self.spatblock5 = BasicBlock2(1, intermediate_feature) if input_feature>=5 else None
        self.spatblock6 = BasicBlock2(1, intermediate_feature) if input_feature>=6 else None
        self.spatblock7 = BasicBlock2(1, intermediate_feature) if input_feature>=7 else None
        self.spatblock8 = BasicBlock2(1, intermediate_feature) if input_feature>=8 else None
        #self.spatblocks = []
        #for idx in range(input_feature):
        #    self.spatblocks.append(BasicBlock2(1, intermediate_feature))
        self.tempblock = BasicBlock2(input_feature, intermediate_feature)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = x.clone()
        out[:,0:1,...] = self.spatblock1(x[:,0:1,...]) if self.input_feature>=1 else None
        out[:,1:2,...] = self.spatblock2(x[:,1:2,...]) if self.input_feature>=2 else None
        out[:,2:3,...] = self.spatblock3(x[:,2:3,...]) if self.input_feature>=3 else None
        out[:,3:4,...] = self.spatblock4(x[:,3:4,...]) if self.input_feature>=4 else None
        out[:,4:5,...] = self.spatblock5(x[:,4:5,...]) if self.input_feature>=5 else None
        out[:,5:6,...] = self.spatblock6(x[:,5:6,...]) if self.input_feature>=6 else None
        out[:,6:7,...] = self.spatblock7(x[:,6:7,...]) if self.input_feature>=7 else None
        out[:,7:8,...] = self.spatblock8(x[:,7:8,...]) if self.input_feature>=8 else None
 
        #for idx in range(x.size()[1]):
        #    out.append(self.spatblocks[idx](x[:,idx:idx+1,...]))
        #out = torch.cat(out,1)
        out = self.tempblock(out)
        return out


class MultipleBasicBlock(nn.Module):

    def __init__(self,input_feature,
                 block, num_blocks,
                 intermediate_feature = 64, dense = True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature

        self.cvlayer1 = nn.Sequential(*[nn.Conv2d(input_feature, intermediate_feature*2,
                      kernel_size=9, stride=1, padding=4, bias=True),nn.LeakyReLU(inplace=True)])
        self.cvlayer2 = nn.Sequential(*[nn.Conv2d(intermediate_feature*2, intermediate_feature,
                      kernel_size=5, stride=1, padding=2, bias=True),nn.LeakyReLU(inplace=True)])
        self.block1 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=1 else None
        self.block2 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=2 else None
        self.block3 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=3 else None
        self.block4 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=4 else None
        self.block5 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=5 else None
        #self.block6 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=6 else None
        #self.block7 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=7 else None
        #self.block8 = nn.Sequential(*[nn.Conv2d(intermediate_feature, 1 , (3, 3), 1, (1, 1)),nn.Sigmoid()])
        self.BN1 = nn.BatchNorm2d(intermediate_feature) if num_blocks>=1 else None
        self.BN2 = nn.BatchNorm2d(intermediate_feature) if num_blocks>=2 else None
        self.BN3 = nn.BatchNorm2d(intermediate_feature) if num_blocks>=3 else None
        self.BN4 = nn.BatchNorm2d(intermediate_feature) if num_blocks>=4 else None
        self.BN5 = nn.BatchNorm2d(intermediate_feature) if num_blocks>=5 else None
        self.do1 = nn.Dropout2d(p=0.2) if num_blocks>=1 else None
        self.do2 = nn.Dropout2d(p=0.2) if num_blocks>=2 else None
        self.do3 = nn.Dropout2d(p=0.2) if num_blocks>=3 else None
        self.do4 = nn.Dropout2d(p=0.2) if num_blocks>=4 else None
        self.do5 = nn.Dropout2d(p=0.2) if num_blocks>=5 else None
        #self.BN7     = nn.BatchNorm2d(intermediate_feature)
        self.BNend     = nn.BatchNorm2d(intermediate_feature)
        self.endlayer = nn.Sequential(*[nn.Conv2d(intermediate_feature, 1 , 5, 1, 2),nn.Sigmoid()])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.do1(x)
        x = self.cvlayer1(x)
        x = self.cvlayer2(x)
        x = self.do1(x)
        x = self.BN1(x)
        x = self.block1(x)
        x = self.do2(x)
        x = self.BN2(x)
        x = self.block2(x)
        x = self.do3(x)    if self.num_block>=3  else x
        x = self.BN3(x)    if self.num_block>=3  else x
        x = self.block3(x) if self.num_block>=3 else x
        x = self.do4(x)    if self.num_block>=4  else x
        x = self.BN4(x)    if self.num_block>=4  else x
        x = self.block4(x) if self.num_block>=4 else x
        x = self.do5(x)    if self.num_block>=5  else x
        x = self.BN5(x)    if self.num_block>=5  else x
        x = self.block5(x) if self.num_block>=5 else x
        #x = self.BN5(x)     if self.num_block>5  else x
        #x = self.block6(x) if self.num_block>=6 else x
        #x = self.BN6(x)     if self.num_block>6  else x
        #x = self.block7(x) if self.num_block>=7 else x
        #x = self.block8(x)
        x = self.BNend(x)
        x = self.endlayer(x)
        return x

def MultipleCascadeBlock_func(intermediate_feature = 64):
    model = MultipleCascadeBlock(intermediate_feature)
    return model

def MultipleBasicBlock_4(input_feature,intermediate_feature = 64, num_blocks=4):
    model = MultipleBasicBlock(input_feature,
                               BasicBlock,num_blocks ,
                               intermediate_feature)
    return model


if __name__ == '__main__':

    # x= Variable(torch.randn(2,3,224,448))
    # model =    S2DF(BasicBlock,3,True)
    # y = model(x)
    model = MultipleBasicBlock(200, BasicBlock,4)
    model = BasicBlock(64,64,1)
    # y = model(x)
    exit(0)
