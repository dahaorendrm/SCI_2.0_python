import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as weight_init
import torch
__all__ = ['MultipleBasicBlock','MultipleBasicBlock_4']
def conv3x3(in_planes, out_planes, dilation = 1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.do = nn.Dropout2d(p=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.do(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)

        return out

class BeginBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BeginBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes,
                      kernel_size=9, stride=stride, padding=4, bias=True)
        self.conv2 = nn.Conv2d(planes, planes/2,
                      kernel_size=5, stride=stride, padding=2, bias=True)
        self.bn = nn.BatchNorm2d(planes/2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.do = nn.Dropout2d(p=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.do(x)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu2(out)
        return out

class EndBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(EndBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.endlayer = nn.Conv2d(inplanes, out_channels=1 , kernel_size=5, strid=1, padding=2)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.endlayer(out)
        out = self.sigmoid(out)

        #out += residual
        #out = self.relu(out)

        return out

class MultipleBasicBlock(nn.Module):

    def __init__(self,input_feature,
                 block, num_blocks,
                 intermediate_feature = 64, dense = True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature


        # for i in range(1, num_blocks):
        self.beginblock = BeginBlock(input_feature, intermediate_feature*2, dilation = 1)
        self.block2 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=2 else None
        self.block3 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=3 else None
        self.block4 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=4 else None
        self.block5 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=5 else None
        #self.block6 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=6 else None
        #self.block7 = block(intermediate_feature, intermediate_feature, dilation = 1) if num_blocks>=7 else None
        #self.block8 = nn.Sequential(*[nn.Conv2d(intermediate_feature, 1 , (3, 3), 1, (1, 1)),nn.Sigmoid()])
        self.endlayer = nn.Sequential(*[nn.Conv2d(intermediate_feature, 1 , 5, 1, 2),nn.Sigmoid()])
        #self.BN2     = nn.BatchNorm2d(intermediate_feature)
        #self.BN3     = nn.BatchNorm2d(intermediate_feature)
        #self.BN4     = nn.BatchNorm2d(intermediate_feature)
        #self.BN5     = nn.BatchNorm2d(intermediate_feature)
        #self.BN6     = nn.BatchNorm2d(intermediate_feature)
        #self.BN7     = nn.BatchNorm2d(intermediate_feature)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.beginblock(x)
        x = self.block2(x) if self.num_block>=2 else x
        #x = self.BN2(x)     if self.num_block>2  else x
        x = self.block3(x) if self.num_block>=3 else x
        #x = self.BN3(x)     if self.num_block>3  else x
        x = self.block4(x) if self.num_block>=4 else x
        #x = self.BN4(x)     if self.num_block>4  else x
        x = self.block5(x) if self.num_block>=5 else x
        #x = self.BN5(x)     if self.num_block>5  else x
        #x = self.block6(x) if self.num_block>=6 else x
        #x = self.BN6(x)     if self.num_block>6  else x
        #x = self.block7(x) if self.num_block>=7 else x
        #x = self.block8(x)
        x = self.endlayer(x)
        return x

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
