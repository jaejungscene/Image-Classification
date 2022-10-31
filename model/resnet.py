import math
import torch
import torch.nn as nn
from module import Bottleneck, BasicBlock


__model__ = [
    # resnet
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",

    # resnext
    "resnext18",
    "resnext34",
    "resnext50",
    "resnext101",
    "resnext152",
    "resnext200",

    # se-resnet
    "se-resnet18",
    "se-resnet34",
    "se-resnet50",
    "se-resnet101",
    "se-resnet152",
    "se-resnet200",

    # se-resnext
    "se-resnext18",
    "se-resnext34",
    "se-resnext50",
    "se-resnext101",
    "se-resnext152",
    "se-resnext200",
    
    # cbam-resnet
    "cbam-resnet18",
    "cbam-resnet34",
    "cbam-resnet50",
    "cbam-resnet101",
    "cbam-resnet152",
    "cbam-resnet200",

    # cbam-resnext
    "cbam-resnext18",
    "cbam-resnext34",
    "cbam-resnext50",
    "cbam-resnext101",
    "cbam-resnext152",
    "cbam-resnext200",
]


def create_model(model_name, args):
    model = getattr('ResNet')('cifar', 50,100,224)
    return model


class ResNet(nn.Module):
    def __init__(   
                    self,
                    depth:int,
                    num_classes,
                    imgSize,
                    bottleneck=False,
                    se=False,
                    cbam=False,
                ):
        super(ResNet, self).__init__()        
        self.imgSize = imgSize # input size
        self.se = se # choosing activating the seUnit
        self.cbam = cbam # choosing activating the CBAM

        if imgSize==32:
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
        if self.imgSize==32:
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