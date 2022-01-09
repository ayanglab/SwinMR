# SwinMR 

This is the official implementation of our proposed SwinMR:

xxx

Please cite:

xxx


## Requirements

matplotlib==3.3.4

opencv-python==4.5.3.56

Pillow==8.3.2

pytorch-fid==0.2.0

scikit-image==0.17.2

scipy==1.5.4

tensorboardX==2.4

timm==0.4.12

torch==1.9.0

torchvision==0.10.0

## Training and Testing
Use different options (json files) to train different networks.

### Calgary Campinas multi-channel dataset (CC) 

To train SwinMR (PI) on CC:

`python main_train_swinmr.py --opt ./options/train_swinmr_pi.json`

To test SwinMR (PI) on CC:

`python main_train_swinmr.py --opt ./options/train_swinmr_npi.json`

To train SwinMR (nPI) on CC:

`python main_test_swinmr.py --opt ./options/test/test_swinmr_pi.json`

To test SwinMR (nPI) on CC:

`python main_test_swinmr.py --opt ./options/test/test_swinmr_npi.json`

### Multi-modal Brain Tumour Segmentation Challenge 2017 (BraTS17)

To train SwinMR (nPI) on BraTS17:

`python main_train_swinmr.py --opt ./options/train_swinmr_brats17.json`

To test SwinMR (nPI) on BraTS17:

`python main_test_swinmr_BraTS17.py --opt ./options/test/test_swinmr_brats17.json`



This repository is based on:

SwinIR: Image Restoration Using Swin Transformer ([code](https://github.com/JingyunLiang/SwinIR) and 
[paper](https://arxiv.org/abs/2108.10257));

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
([code](https://github.com/microsoft/Swin-Transformer) and [paper](https://arxiv.org/abs/2103.14030)).