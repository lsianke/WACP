## Environments
The code has been tested in the following environments:

- Python 3.8  
- PyTorch 1.8.1
- cuda 10.2
- torchsummary, torchvision, thop, scipy, sympy
## Pre-trained Models
[VGG-16](https://drive.google.com/file/d/1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE/view) | [GoogLeNet](https://drive.google.com/file/d/1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c/view) | [ResNet-56](https://drive.google.com/file/d/1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T/view)  | [ResNet-50](https://drive.google.com/file/d/1OYpVB84BMU0y-KU7PdEPhbHwODmFvPbB/view) | [MobileNet-V2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) 
## Running Code
The settings are listed below:
1. VGG-16

|       Compression Rate        | FLOPs($\downarrow$) | Paeameters($\downarrow$) | Accuracy(%) |
| :---------------------------: | :-----------------: | :----------------------: | :---------: |
| [0.25]\*5+[0.35]\*3+[0.75]\*5 |   118.48M/(63.2%)   |      2.40M/(84.2%)       |   93.86%    |
| [0.35]\*5+[0.45]\*3+[0.8]\*5  |   88.04M/(71.9%)    |      1.69M/(88.7%)       |   93.53%    |
```
#VGG-16
#All run scripts can be cut-copy-paste from run.bat or run.sh.
python main.py \
--arch vgg_16_bn \
--resume [pre-trained model dir] \
--compress_rate [0.35]*5+[0.45]*3+[0.8]*5 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch vgg_16_bn \
--from_scratch True \
--resume final_pruned_model/vgg_16_bn_1.pt \
--num_workers 1 \
--epochs 200 \
--gpu 0 \
--lr 0.01 \
--lr_decay_step 100,150 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

2. GoogLeNet

|           Compression Rate            | FLOPs($\downarrow$) | Paeameters($\downarrow$) | Accuracy(%) |
| :-----------------------------------: | :-----------------: | :----------------------: | :---------: |
| [0.2]+[0.7]\*15+[0.75]*\9+[0.,0.4,0.] |   0.95B/(61.5%)     |      2.86M/(53.5%)       |   95.16%    |
|      [0.2]+[0.9]\*24+[0.,0.4,0.]      |   0.38B/(75.2%)     |      2.21M/(64.1%)       |   94.64%    |

```
python main.py \
--arch googlenet \
--resume [pre-trained model dir] \
--compress_rate [0.2]+[0.9]*24+[0.,0.3,0.] \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch googlenet \
--from_scratch True \
--resume final_pruned_model/googlenet_1.pt \
--num_workers 1 \
--epochs 200 \
--lr 0.01 \
--lr_decay_step 100,150 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```
3. ResNet-56

|                       Compression Rate                       | FLOPs($\downarrow$) | Paeameters($\downarrow$) | Accuracy(%) |
| :----------------------------------------------------------: | :-----------------: | :----------------------: | :---------: |
| [0.]+[0.,0.]\*1+[0.5,0.]\*8+[0.,0.]\*1+[0.5,0.]\*8+[0.,0.]\*1+[0.2,0.]\*8 |    81.5M/(35.1%)    |      0.65M/(23.5%)       |   94.02%    |
| [0.]+[0.2,0.]\*1+[0.65,0.]\*8+[0.2,0.15]\*1+[0.65,0.15]\*8+[0.2,0.]\*1+[0.4,0.]\*8 |    60.0M/(52.2%)    |      0.48M/(43.5%)       |   93.72%    |
| [0.]+[0.3,0.2]\*1+[0.7,0.2]\*8+[0.3,0.2]\*1+[0.7,0.2]\*8+[0.3,0.2]\*1+[0.4,0.2]\*8 |    46.3M/(63.1%)    |      0.39M/(54.1%)       |   93.17%    |

```
python main.py \
--arch resnet_56 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.3,0.2]\*1+[0.7,0.2]\*8+[0.3,0.2]\*1+[0.7,0.2]\*8+[0.3,0.2]\*1+[0.4,0.2]\*8 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch resnet_56 \
--from_scratch True \
--resume final_pruned_model/resnet_56_1.pt \
--num_workers 1 \
--epochs 300 \
--gpu 0 \
--lr 0.01 \
--lr_decay_step 150,225 \
--weight_decay 0.0005 \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

4. ResNet-50

|                       Compression Rate                       | FLOPs($\downarrow$) | Accuracy(%) |
| :----------------------------------------------------------: | :-----------------: | :----------------------: | :---------: |
| [0.]+[0.2,0.2,0.1]\*1+[0.65,0.65,0.1]\*2+[0.2,0.2,0.15]\*1+[0.65,0.65,0.15]\*3+[0.2,0.2,0.15]\*1+[0.65,0.65,0.15]\*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]\*2 |   1.83B/(55.3%)   |       75.45%/92.45%    |
| [0.]+[0.2,0.2,0.2]\*1+[0.75,0.75,0.2]\*2+[0.2,0.2,0.2]\*1+[0.75,0.75,0.2]\*3+[0.2,0.2,0.2]\*1+[0.75,0.75,0.2]\*5+[0.2,0.2,0.2]+[0.2,0.2,0.2]\*2 |   1.59B/(61.3%)    |         74.75%/92.19%    |

```
python main.py \
--arch resnet_50 \
--resume [pre-trained model dir] \
--data_dir [dataset dir] \
--dataset ImageNet \
--compress_rate [0.]+[0.2,0.2,0.2]\*1+[0.75,0.75,0.2]\*2+[0.2,0.2,0.2]\*1+[0.75,0.75,0.2]\*3+[0.2,0.2,0.2]\*1+[0.75,0.75,0.2]\*5+[0.2,0.2,0.2]+[0.2,0.2,0.2]\*2\
--num_workers 4 \
--batch_size 128 \
--epochs 1 \
--lr_decay_step 1 \
--lr 0.001 \
--weight_decay 0. \
--input_size 224 \
--save_id 1 

python main.py \
--arch resnet_50 \
--from_scratch True \
--resume finally_pruned_model/resnet_50_1.pt \
--num_workers 4 \
--epochs 120 \
--lr 0.01 \
--lr_decay_step 30,60,90 \
--batch_size 128 \
--weight_decay 0.0001 \
--input_size 224 \
--data_dir [dataset dir] \
--dataset ImageNet \
--save_id 1
```
5. MobileNet-V2

|                       Compression Rate                       | FLOPs($\downarrow$) |  Accuracy(%) |
| :----------------------------------------------------------: | :-----------------: | :----------------------: | :---------: |
| [0.]+ [0.1]\*2+[0.1]\*2+[0.3]+[0.1]\*2+[0.3]\*2+[0.1]\*2+[0.3]\*3+[0.1]\*2+[0.3]\*2+[0.1]\*2+[0.3]\*2+ [0.1]\*2+[0.2] |   215.31M/(30.6%)   |   71.20%/89.83%    |

```
python main.py \
--arch mobilenet_v2 \
--resume [pre-trained model dir] \
--data_dir [dataset dir] \
--dataset ImageNet \
--compress_rate [0.]+[0.1]*2+[0.1]*2+[0.3]+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*3+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*2+[0.1]*2 \
--num_workers 4 \
--batch_size 128 \
--epochs 1 \
--lr_decay_step 1 \
--lr 0.001 \
--weight_decay 0. \
--input_size 224 \
--save_id 1 

python main.py \
--arch resnet_50 \
--from_scratch True \
--resume finally_pruned_model/mobilenet_v2_1.pt \
--num_workers 4 \
--epochs 150 \
--lr 0.01 \
--lr_decay_step 30,60,90 \
--batch_size 128 \
--weight_decay 0.0001 \
--input_size 224 \
--data_dir [dataset dir] \
--dataset ImageNet \
--save_id 1
```
## Acknowledgments
Our implementation partially reuses [HRank's](https://github.com/lmbxmu/HRank)
