import math
import torch
import torch.nn as nn


def conv3x3(in_channel, out_channel, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, se=False, cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEblock(planes) if se else nn.Identity()
        self.cbam = CBAM(planes) if cbam else nn.Identity()

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

        out = self.se(out)
        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, se=False, cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEblock(planes * Bottleneck.expansion) if se else nn.Identity()
        self.cbam = CBAM(planes * Bottleneck.expansion) if cbam else nn.Identity()

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

        out = self.se(out)
        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True, norm_layer=nn.BatchNorm2d, act=True):
        super(ConvNormAct,self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(out_ch) if norm_layer != nn.Identity() else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )



class SEblock(nn.Sequential):
    def __init__(self, channel, r=16):
        super(SEblock, self).__init__(
            # squeeze
            nn.AdaptiveAvgPool2d(1), 

            # excitation
            ConvNormAct(channel, channel//r, 1),
            nn.Conv2d(channel//r, channel, 1, bias=True),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        out = super(SEblock, self).forward(x)
        return x + out



class CBAM(nn.Module):
    def __init__(self, channel, r=16):
        super(CBAM, self).__init__()
        self.avg_channel = nn.AdaptiveAvgPool2d(1)
        self.max_channel = nn.AdaptiveMaxPool2d(1)
        self.shared_excitation = nn.Sequential(
            ConvNormAct(channel, channel//r, 1, bias=False, norm_layer=nn.Identity),
            nn.Conv2d(channel//r, channel, 1, bias=False)
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        ## channel attention ##
        out1 = self.avg_channel(input)
        out1 = self.shared_excitation(out1)
        out2 = self.max_channel(input)
        out2 = self.shared_excitation(out2)
        channel_attention = nn.Sigmoid()(out1+out2) # (batch, channel, 1, 1)
        input = input * channel_attention

        ## spatial attention ##
        batch, size,_,_ = input.shape
        avg_spatial = input.mean(dim=1).reshape(batch, 1, size, -1) # (batch, 1, H, W)
        max_spatial = input.max(dim=1)[0].reshape(batch, 1, size, -1) # (batch, 1, H, W)
        spatial_attention = torch.cat([avg_spatial, max_spatial], 1)
        spatial_attention = self.conv_spatial(spatial_attention)
        input = input * spatial_attention

        return input



class ResNet(nn.Module):
    def __init__(   
                    self,
                    dataset,
                    depth,
                    num_classes,
                    insize,
                    bottleneck=False,
                    se=False,
                    cbam=False,
                ):
        super(ResNet, self).__init__()        
        self.dataset = dataset # type of dataset
        self.insize = insize # input size
        self.se = se # choosing activating the seUnit
        self.cbam = cbam # choosing activating the CBAM

        if self.dataset.startswith('cifar') and insize==32: # if dataset is cifar...
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2) 
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        else: # if dataset is imagenet, or input size is (224,224) in cifar
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): # He initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # normal distribution parameterized by mean=0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, se=self.se, cbam=self.cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se, cbam=self.cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.insize==32:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
        return x
