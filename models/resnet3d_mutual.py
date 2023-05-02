import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import pdb
import math
from .mutual_attention import MAM
from functools import partial
from tools.helplayer import BNClassifier , BottleSoftmax , weights_init_kaiming , weights_init_classifier
from .seNet import *
from .resnet_ibn_a import *
from .cam import CAM

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]



#def get_attention(self, a):
 #       input_a = a

  #      a = a.mean(3) 
   #     a = a.transpose(1, 3) 
     #   a = F.relu(self.conv1(a))
      #  a = self.conv2(a) 
      #  a = a.transpose(1, 3)
       # a = a.unsqueeze(3) 
        
       # a = torch.mean(input_a * a, -1) 
       # a = F.softmax(a / 0.025, dim=-1) + 1
       # return a 


class Attention_mod(nn.Module):
    def __init__(self): #flow thresh selected base on avg flow -std dev in dataset
        super(Attention_mod, self).__init__()
       
       
        self.attention = None
       

    def forward(self, x,flow_a):

       
        attention_f = torch.mean(flow_a, dim=1, keepdim=True)
        attention_x = torch.mean(x, dim=1, keepdim=True)       
       
        mutual = attention_x * attention_f
        # Generate importance map
        importance_map = torch.sigmoid(mutual)

      
#        pdb.set_trace()
        self.attention = importance_map
        output = x.mul(importance_map)

        return output

    def get_maps(self):
        return self.attention, self.drop_mask


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_height,
                 sample_width,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)


#        resnet50 = torchvision.models.resnet50(pretrained=True)
#        resnet50 = se_resnet50(pretrained=True)
        resnet50 = resnet50_ibn_a(last_stride=1, pretrained=True)             

#        resnet50.layer4[0].conv2.stride = (1,1) # uncomment for RESNET
 #       resnet50.layer4[0].downsample[0].stride = (1,1) # uncomment for RESNET
#        self.base_a = nn.Sequential(*list(resnet50.children())[:-2])
        layers_list = list(resnet50.children())[:-2] ## 2
       # pdb.set_trace() 
        self.base_a1 = nn.Sequential(*layers_list[0:7]) ## 0 to 7
        self.base_a2 = nn.Sequential(*layers_list[7:8]) ## 7 to 8

        self.attention = CAM () #Attention_mod()


#        self.feat_dim = 4096
#        self.mam = MAM()
   #     self.RELU =  nn.ReLU()
#        self.interpol = Interpolate(size=(2048, 1), mode='bilinear')  
#        self.fag_embedding_init = nn.Linear(2048, 2048)
 #       self.fag_embedding = nn.Linear(2048, 2048)
  #      self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.fc = BNClassifier(2048, 625 , initialization=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def load_matched_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            #if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            param = param.data
            print("loading "+name)
            own_state[name].copy_(param)

    def forward(self, x):
        # default size is (b, s, c, w, h), s for seq_len, c for channel
        # convert for 3d cnn, (b, c, s, w, h)
        b = x.size(0)
        t = x.size(1)
        x_a = x

        x_a = x_a.view(b*t,x.size(2), x.size(3), x.size(4))
#        pdb.set_trace()       
        x_p1 = self.base_a1(x_a)
       
        x=x.permute(0,2,1,3,4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3d_l3 = self.layer3(x)

        if x3d_l3.size(2)>1:
           x3d_l3 = torch.mean(x3d_l3,2,keepdim =True)  


        x_p2 = x_p1.view(b,t,x_p1.size(1),x_p1.size(2),x_p1.size(3)) #1024,24,8)

#        x_k = x_p2.permute(0,2,1,3,4)
        x3d = x3d_l3.permute(0,2,1,3,4)
#        pdb.set_trace()       
        
        x_att_1,x_att2 = self.attention(x_p2,x3d)
        
 #       x_att = self.mam(x_p1,x3d_l3,b,t)
#        pdb.set_trace()   

        #x_att_2 = x_att_1.permute(0,2,1,3,4)        
        x_att_1 = x_att2+x_att_1
        x_att = x_att_1.view(b*t,x_p1.size(1),x_p1.size(2),x_p1.size(3)) # 1024,24,8)
        #pdb.set_trace()                
        x_att = x_att + x_p1       
        x_att = F.normalize(x_att, dim =1)

#        pdb.set_trace()
        x_p1 = self.base_a2(x_att)
        x_p = F.avg_pool2d(x_p1, x_p1.size()[2:])



        x_p = x_p.view(b,t,-1)
        x_p=x_p.permute(0,2,1)
        x_f = F.avg_pool1d(x_p,t)
        x_f = x_f.view(b, 2048)
       

        f,y = self.fc(x_f)

        return y,f


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break

        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
