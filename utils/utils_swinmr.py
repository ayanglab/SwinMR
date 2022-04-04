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


# def load_images(H_path, SM_path, isSM=True):
#     # load GT
#     gt = np.load(H_path).astype(np.float32)
#
#     gt = np.reshape(gt, (gt.shape[0], gt.shape[1], 1))
#     # # 0 ~ 255
#     gt = (gt - gt.min()) / (gt.max() - gt.min()) * 255
#
#     # load SM
#     if isSM == True:
#         sm = np.load(SM_path).astype(np.float32)[:, :, :, 1]
#
#         # sm = np.reshape(sm[:, :, :, 1], (256, 256, 12))
#
#         # 0 ~ 1
#         sm = (sm - sm.min()) / (sm.max() - sm.min())
#
#         return gt, sm
#     else:
#         return gt, 0


# def save_img(imgs, savedir):
#     # for sample
#     img_weight = imgs[0, :, :, 0:1]
#     sm = imgs[0, :, :, 1:imgs.shape[3]]
#
#     img_weight = (img_weight + 1) * 127.5
#     img = img_weight * sm
#     sm = sm * 255
#
#     img_weight = img_weight.astype(np.uint8)
#     cv2.imwrite(os.path.join(savedir, 'img_weight.png'), img_weight)
#
#     img = img.astype(np.uint8)
#     for i in range(imgs.shape[3] - 1):
#         cv2.imwrite(os.path.join(savedir, 'img_{}.png'.format(i)), img[:, :, i:i + 1])
#
#     sm = sm.astype(np.uint8)
#     for i in range(imgs.shape[3] - 1):
#         cv2.imwrite(os.path.join(savedir, 'sm_{}.png'.format(i)), sm[:, :, i:i + 1])


# def undersample_kspace(x, mask, is_noise, noise_level, noise_var):
#
#     fft = fft2(x[:, :, 0])
#     fft = fftshift(fft)
#     fft = fft * mask
#     if is_noise:
#         fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)
#     fft = ifftshift(fft)
#     xx = ifft2(fft)
#     xx = np.abs(xx)
#
#     x = xx[:, :, np.newaxis]
#
#     return x
#
# def generate_gaussian_noise(x, noise_level, noise_var):
#     spower = np.sum(x ** 2) / x.size
#     npower = noise_level / (1 - noise_level) * spower
#     noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
#     return noise

# # Sobel
# def sobel(src, device):
#
#     # define convolution
#     conv_opx = nn.Conv2d(src.shape[1], src.shape[1], kernel_size=3, padding=1, bias=False).to(device)
#     conv_opy = nn.Conv2d(src.shape[1], src.shape[1], kernel_size=3, padding=1, bias=False).to(device)
#
#     # define sobel kernel
#     sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
#     sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
#     sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
#     sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
#     # set kernel channel
#     sobel_kernel_x = np.repeat(np.repeat(sobel_kernel_x, src.shape[1], axis=0), src.shape[1], axis=1)
#     sobel_kernel_y = np.repeat(np.repeat(sobel_kernel_y, src.shape[1], axis=0), src.shape[1], axis=1)
#
#     # load conv kernel
#     conv_opx.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
#     conv_opy.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
#
#     dst_x = conv_opx(src)
#     dst_y = conv_opy(src)
#     dst = torch.abs(torch.add(dst_x/2, dst_y/2))
#
#     # 0~+
#     return dst