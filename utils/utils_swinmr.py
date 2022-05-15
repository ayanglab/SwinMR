import torch
from torch import nn
import os
import cv2
import gc
import numpy as np
from scipy.io import *
from scipy.fftpack import *


"""
# --------------------------------------------
# Jiahao Huang (j.huang21@imperial.uk.ac)
# 30/Jan/2022
# --------------------------------------------
"""


# Fourier Transform
def fft_map(x):
    fft_x = torch.fft.fftn(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag

    return fft_x_real, fft_x_imag

