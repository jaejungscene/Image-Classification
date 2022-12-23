import torch
from .resnet import ResNet

def get_model(args):
    model = ResNet(args.depth, args.num_classes, args.imgSize)
    return model 