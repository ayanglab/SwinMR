import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.fft


"""
# --------------------------------------------
# Sobel Filter
# --------------------------------------------
# Jiahao Huang (j.huang21@imperial.uk.ac)
# 30/Jan/2022
# --------------------------------------------
"""


# Sobel
def sobel(src, device):

    src = torch.clamp(src, 0, 1)

    # define convolution
    conv_opx = nn.Conv2d(src.shape[1], src.shape[1], kernel_size=3, padding=1, bias=False).to(device)
    conv_opy = nn.Conv2d(src.shape[1], src.shape[1], kernel_size=3, padding=1, bias=False).to(device)

    # define sobel kernel
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
    sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
    # set kernel channel
    sobel_kernel_x = np.repeat(np.repeat(sobel_kernel_x, src.shape[1], axis=0), src.shape[1], axis=1)
    sobel_kernel_y = np.repeat(np.repeat(sobel_kernel_y, src.shape[1], axis=0), src.shape[1], axis=1)

    # load conv kernel
    conv_opx.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
    conv_opy.weight.data = torch.from_numpy(sobel_kernel_y).to(device)


    dst_x = conv_opx(src)
    dst_y = conv_opy(src)
    dst = torch.abs(torch.add(dst_x/2, dst_y/2))

    # 0~+
    return dst

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
    x_sobel = sobel(x, device)
    x_sobel = x_sobel.cpu().squeeze().detach().numpy()
    print(x_sobel.shape)
    x_sobel = x_sobel[slice_idx, :, :]

    cv2.imwrite("../tmp/Sobel.png", 255 * (x_sobel-x_sobel.min())/(x_sobel.max()-x_sobel.min()))


