'''
# -----------------------------------------
Define Undersampling Mask
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import os
import scipy
import scipy.fftpack
from scipy.io import loadmat
import cv2
import numpy as np


def define_Mask(opt):
    mask_name = opt['mask']

    # 256 * 256 Gaussian 1D
    if mask_name == 'G1D10':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_10.mat"))['maskRS1']
    elif mask_name == 'G1D20':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_20.mat"))['maskRS1']
    elif mask_name == 'G1D30':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_30.mat"))['maskRS1']
    elif mask_name == 'G1D40':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_40.mat"))['maskRS1']
    elif mask_name == 'G1D50':
        mask = loadmat(os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_50.mat"))['maskRS1']

    # 256 * 256 Gaussian 2D
    elif mask_name == 'G2D10':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_10.mat"))['maskRS2']
    elif mask_name == 'G2D20':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_20.mat"))['maskRS2']
    elif mask_name == 'G2D30':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_30.mat"))['maskRS2']
    elif mask_name == 'G2D40':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_40.mat"))['maskRS2']
    elif mask_name == 'G2D50':
        mask = loadmat(os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_50.mat"))['maskRS2']

    # 256 * 256 poisson 2D
    elif mask_name == 'P2D10':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_10.mat"))['population_matrix']
    elif mask_name == 'P2D20':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_20.mat"))['population_matrix']
    elif mask_name == 'P2D30':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_30.mat"))['population_matrix']
    elif mask_name == 'P2D40':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_40.mat"))['population_matrix']
    elif mask_name == 'P2D50':
        mask = loadmat(os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_50.mat"))['population_matrix']

    # 256 * 256 radial
    elif mask_name == 'R10':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_10.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R20':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_20.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R30':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_30.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R40':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_40.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R50':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_50.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R60':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_60.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R70':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_70.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R80':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_80.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'R90':
        mask_shift = cv2.imread(os.path.join('mask', 'radial', 'radial_90.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)

    # 256 * 256 spiral
    elif mask_name == 'S10':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_10.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S20':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_20.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S30':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_30.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S40':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_40.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S50':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_50.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S60':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_60.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S70':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_70.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S80':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_80.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == 'S90':
        mask_shift = cv2.imread(os.path.join('mask', 'spiral', 'spiral_90.tif'), 0) / 255
        mask = scipy.fftpack.fftshift(mask_shift)

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(mask_name))

    print('Training model [{:s}] is created.'.format(mask_name))

    return mask
