# SwinMR 

by Jiahao Huang (j.huang21@imperial.ac.uk)

This is the official implementation of our proposed SwinMR:

Swin Transformer for Fast MRI

Please cite:

```
@article{HUANG2022281,
    title = {Swin transformer for fast MRI},
    journal = {Neurocomputing},
    volume = {493},
    pages = {281-304},
    year = {2022},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2022.04.051},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231222004179},
    author = {Jiahao Huang and Yingying Fang and Yinzhe Wu and Huanjun Wu and Zhifan Gao and Yang Li and Javier Del Ser and Jun Xia and Guang Yang},
    keywords = {MRI reconstruction, Transformer, Compressed sensing, Parallel imaging},
    abstract = {Magnetic resonance imaging (MRI) is an important non-invasive clinical tool that can produce high-resolution and reproducible images. However, a long scanning time is required for high-quality MR images, which leads to exhaustion and discomfort of patients, inducing more artefacts due to voluntary movements of the patients and involuntary physiological movements. To accelerate the scanning process, methods by k-space undersampling and deep learning based reconstruction have been popularised. This work introduced SwinMR, a novel Swin transformer based method for fast MRI reconstruction. The whole network consisted of an input module (IM), a feature extraction module (FEM) and an output module (OM). The IM and OM were 2D convolutional layers and the FEM was composed of a cascaded of residual Swin transformer blocks (RSTBs) and 2D convolutional layers. The RSTB consisted of a series of Swin transformer layers (STLs). The shifted windows multi-head self-attention (W-MSA/SW-MSA) of STL was performed in shifted windows rather than the multi-head self-attention (MSA) of the original transformer in the whole image space. A novel multi-channel loss was proposed by using the sensitivity maps, which was proved to reserve more textures and details. We performed a series of comparative studies and ablation studies in the Calgary-Campinas public brain MR dataset and conducted a downstream segmentation experiment in the Multi-modal Brain Tumour Segmentation Challenge 2017 dataset. The results demonstrate our SwinMR achieved high-quality reconstruction compared with other benchmark methods, and it shows great robustness with different undersampling masks, under noise interruption and on different datasets. The code is publicly available at https://github.com/ayanglab/SwinMR.}
}

```

![Overview_of_SwinMR](./tmp/files/SwinMR.png)


## Highlight

- A novel Swin transformer-based model for fast MRI reconstruction was proposed.
- A multi-channel loss with sensitivity maps was proposed for reserving more details.
- Comparison studies were performed to validate the robustness of our SwinMR.
- A pre-trained segmentation network was used to validate the reconstruction quality.


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

