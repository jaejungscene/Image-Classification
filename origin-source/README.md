# pytorch-cifar-example
This repo contains pytorch cifar training examples. Main goal of this repo is to provide reference material for basic pytorch classification techniques. This repo is highly affected by timm and torchvision reference. 

Techniques used in this repo

| No   | Technique                    | Type         | Code                             | Reference                                                    |
| ---- | ---------------------------- | ------------ | -------------------------------- | ------------------------------------------------------------ |
| 1    | Distributed data parallel    | multi-gpu    | [setup.py](setup.py)             | [ddp docs](https://pytorch.org/docs/stable/elastic/run.html), [torchvsion reference code](https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L245) |
| 2    | Auto / Rand / TA aug.        | augmentation | [dataset.py](dataset.py#L32)     | [torchvision transforms docs](https://pytorch.org/vision/stable/transforms.html) |
| 3    | Cutmix / Mixup / Cutout      | augmentation | [dataset.py](dataset.py#L112)    | [cutmix](https://github.com/clovaai/CutMix-PyTorch), [cutmix/mixup code](https://github.com/pytorch/vision/blob/main/references/classification/transforms.py) |
| 4    | ResNet / SE-ResNet / ResNext | model        | [resnet.py](resnet.py#L77)       | [resnet paper](https://arxiv.org/abs/1512.03385) [torchvision resnet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) |
| 5    | SGD / AdamW / RMSProp        | optimizer    | [optimizer.py](optimizer.py#L69) |                                                              |
| 6    | Cosine / Warmup / Multistep  | scheduler    | [optimizer.py](optimizer.py#L78) |                                                              |
| 7    | CE / BCE / Label Smoothing   | criterion    | [optimizer.py](optimizer.py#113) |                                                              |
| 8    | Exponential Moving Average   | training     | [utils.py](utils.py#L15)         | [timm ema](https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py) |
| 9    | Auto Mixed Precision (fp16)  | training     | [optimizer.py](optimizer.py#L33) | [pytorch amp docs](https://pytorch.org/docs/stable/amp.html) |
| 10   | Grad Norm / Grad Accumulate  | training     | [train.py](train.py#L110)        | [pytorch grad scaler](https://pytorch.org/docs/stable/amp.html#gradient-scaling) |
| 11   | Logger / Wandb / Multirun    | etc          |                                  |                                                              |



## Tutorial

1. git clone this repo

   ```bash
   git clone https://github.com/Team-Ryu/pytorch-cifar-example.git
   ```

2. run command in belows

   ```bash
   cd pytorch-cifar-example
   torchrun --nproc_per_node=2 --master_port=12345 multi_train.py data cifar10_224 cifar100_224 -m resnet50 resnet110 -c 0,1 -o log/baseline --use-wandb
   ```



## Training Commands

CIFAR10 | Model: ResNet50 | Top-1: 96.61 | Time: 3h 16m

```bash
torchrun --nproc_per_node=2 --master_port=12345 train.py data --dataset_type CIFAR10 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4802 0.4481 0.3975 --std 0.2302 0.2265 0.2262 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last --cuda 0,1
```



CIFAR100 | Model: ResNet50 | Top-1: 82.55 | Time: 3h 15m

```bash
torchrun --nproc_per_node=2 --master_port=12345 train.py data --dataset_type CIFAR100 --train-size 224 224 --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4825 0.4467 --std 0.2471 0.2435 0.2616 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 1 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 4 --pin-memory --amp --channels-last --cuda 0,1
```



Multiple run (CIFAR10, CIFAR100)

```bash
torchrun --nproc_per_node=2 --master_port=12345 multi_train.py data cifar10_224 cifar100_224 -m resnet50 resnet110 -c 0,1 -o log/baseline --use-wandb
```



## Expected Training Results

Cifar10

|           | Top-1  | Top-5  | Train Time | Log                                                          |
| --------- | ------ | ------ | ---------- | ------------------------------------------------------------ |
| ResNet34  | 96.965 | 99.878 | 2h 39m     | [cifar10_r34.txt](https://github.com/Team-Ryu/pytorch-cifar-example/releases/download/v1.0.0/cifar10_r34.txt) |
| ResNet50  | 96.508 | 99.910 | 3h 16m     | [cifar10_r50.txt](https://github.com/Team-Ryu/pytorch-cifar-example/releases/download/v1.0.0/cifar10_r50.txt) |
| ResNet101 | 96.773 | 99.931 | 4h 39m     | [cifar10_r101.txt](https://github.com/Team-Ryu/pytorch-cifar-example/releases/download/v1.0.0/cifar10_r101.txt) |

Cifar100

|           | Top-1  | Top-5  | Train Time | Log                                                          |
| --------- | ------ | ------ | ---------- | ------------------------------------------------------------ |
| ResNet34  | 82.656 | 95.829 | 2h 30m     | [cifar100_r34.txt](https://github.com/Team-Ryu/pytorch-cifar-example/releases/download/v1.0.0/cifar100_r34.txt) |
| ResNet50  | 82.550 | 96.061 | 3h 15m     | [cifar100_r50.txt](https://github.com/Team-Ryu/pytorch-cifar-example/releases/download/v1.0.0/cifar100_r50.txt) |
| ResNet101 | 83.517 | 96.002 | 4h 38m     | [cifar100_r101.txt](https://github.com/Team-Ryu/pytorch-cifar-example/releases/download/v1.0.0/cifar100_r101.txt) |
