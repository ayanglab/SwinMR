import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.fft
from math import pi
import matplotlib.pyplot as plt


"""
# --------------------------------------------
# Gabor Filter
# --------------------------------------------
# Jiahao Huang (j.huang21@imperial.uk.ac)
# 30/Jan/2022
# --------------------------------------------
"""


def gabor(src, device):
    # -1~1 --> 0~1
    src = torch.clamp(src, 0, 1)
    kernel_size = 31
    padding = int((kernel_size - 1) / 2)

    # define convolution
    conv_op0 = nn.Conv2d(src.shape[1], src.shape[1],
                         kernel_size=kernel_size, padding=padding, bias=False, ).to(device)
    conv_op45 = nn.Conv2d(src.shape[1], src.shape[1],
                          kernel_size=kernel_size, padding=padding, bias=False, ).to(device)
    conv_op90 = nn.Conv2d(src.shape[1], src.shape[1],
                          kernel_size=kernel_size, padding=padding, bias=False, ).to(device)
    conv_op135 = nn.Conv2d(src.shape[1], src.shape[1],
                           kernel_size=kernel_size, padding=padding, bias=False, ).to(device)

    # define gabor kernel
    gabor_kernel0 = generate_gabor_kernel(kernel_size, src.shape[1], src.shape[1], 0)
    gabor_kernel45 = generate_gabor_kernel(kernel_size, src.shape[1], src.shape[1], pi / 4)
    gabor_kernel90 = generate_gabor_kernel(kernel_size, src.shape[1], src.shape[1], pi / 2)
    gabor_kernel135 = generate_gabor_kernel(kernel_size, src.shape[1], src.shape[1], 3 * pi / 4)

    # load conv kernel
    conv_op0.weight.data = torch.from_numpy(gabor_kernel0).to(device)
    conv_op45.weight.data = torch.from_numpy(gabor_kernel45).to(device)
    conv_op90.weight.data = torch.from_numpy(gabor_kernel90).to(device)
    conv_op135.weight.data = torch.from_numpy(gabor_kernel135).to(device)

    dst0 = conv_op0(src)
    dst45 = conv_op45(src)
    dst90 = conv_op90(src)
    dst135 = conv_op135(src)
    dst = torch.cat([dst0, dst45, dst90, dst135], 1)
    dst = torch.abs(dst)

    # 0~+
    return dst

# Generate Gabor Kernel
def generate_gabor_kernel(kernel_size, batch_size, channel, degree):
    # define gabor kernel
    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), 2.5, degree, 5, 0.5, psi=0, ktype=cv2.CV_32F)
    gabor_kernel = gabor_kernel.reshape((1, 1, kernel_size, kernel_size))
    # set kernel channel
    gabor_kernel = np.repeat(np.repeat(gabor_kernel, batch_size, axis=0), channel, axis=1)

    return gabor_kernel




if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    slice_idx = 0

    # slice 1 in batch
    x1 = cv2.imread('../tmp/GT_ 1024.png', cv2.IMREAD_GRAYSCALE)
    # (w, h) --> (n, c, w, h)
    x1 = x1[np.newaxis, np.newaxis, :, :]
    x1= x1/255

    # slice 2 in batch
    x2 = cv2.imread('../tmp/GT_1043.png', cv2.IMREAD_GRAYSCALE)
    # (w, h) --> (n, c, w, h)
    x2 = x2[np.newaxis, np.newaxis, :, :]
    x2 = x2/255

    # slice 3 in batch
    x3 = cv2.imread('../tmp/GT_1043.png', cv2.IMREAD_GRAYSCALE)
    # (w, h) --> (n, c, w, h)
    x3 = x3[np.newaxis, np.newaxis, :, :]
    x3 = x3/255

    x = np.concatenate((x1, x2, x3), axis=0)

    #
    # -1~1
    x = torch.Tensor(x).to(device)

    #
    # gabor
    x_gabor = gabor(x, device)
    x_gabor = x_gabor.cpu().squeeze().detach().numpy()
    print(x_gabor.shape)
    x_gabor0 = x_gabor[slice_idx, 0, :, :]
    x_gabor45 = x_gabor[slice_idx, 1, :, :]
    x_gabor90 = x_gabor[slice_idx, 2, :, :]
    x_gabor135 = x_gabor[slice_idx, 3, :, :]

    cv2.imwrite("../tmp/Gabor_0.png", 255 * (x_gabor0-x_gabor0.min())/(x_gabor0.max()-x_gabor0.min()))
    cv2.imwrite("../tmp/Gabor_45.png", 255 * (x_gabor45-x_gabor45.min())/(x_gabor45.max()-x_gabor45.min()))
    cv2.imwrite("../tmp/Gabor_90.png", 255 * (x_gabor90-x_gabor90.min())/(x_gabor90.max()-x_gabor90.min()))
    cv2.imwrite("../tmp/Gabor_135.png", 255 * (x_gabor135-x_gabor135.min())/(x_gabor135.max()-x_gabor135.min()))

    x = x.cpu().squeeze().detach().numpy()
    print(x.shape)
    # show gabor
    plt.suptitle('Gabor')  # 图片名称
    plt.subplot(2, 3, 1), plt.title('GT')
    plt.imshow(x[slice_idx], cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 4), plt.title('GT')
    plt.imshow(x[slice_idx], cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 2), plt.title('0')
    plt.imshow(x_gabor0, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 3), plt.title('pi/4')
    plt.imshow(x_gabor45, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 5), plt.title('pi/2')
    plt.imshow(x_gabor90, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 6), plt.title('3pi/4')
    plt.imshow(x_gabor135, cmap='gray'), plt.axis('off')

    plt.savefig("../tmp/Gabor.png")
    plt.show()


