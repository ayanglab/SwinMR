import scipy
import scipy.fftpack
from scipy.io import loadmat
import numpy as np
import utils.utils_image as util
import os


def main(H_path, L_path, mask):
    isExists = os.path.exists(H_path)
    if not isExists:
        os.makedirs(H_path)
    isExists = os.path.exists(L_path)
    if not isExists:
        os.makedirs(L_path)

    files = os.listdir(H_path)
    # print(files)
    i = 1
    for file in files:
        img_H = util.imread_uint(os.path.join(H_path, file), 1)
        img_L = to_bad_img(img_H, mask)
        util.imsave(img_L, os.path.join(L_path, file))
        print('file:{} {}/{}'.format(os.path.join(L_path, file), i, len(files)))
        i = i + 1

def main_single(H_path, L_path, mask):
    file = 'train_6478.png'
    img_H = util.imread_uint(os.path.join(H_path, file), 1)
    img_L = to_bad_img(img_H, mask)
    util.imsave(img_L, os.path.join(L_path, file))



def to_bad_img(x, mask):
    xx = x[:, :, 0]
    fft = scipy.fftpack.fft2(xx)
    fft = scipy.fftpack.fftshift(fft)
    fft = fft * mask
    fft = scipy.fftpack.ifftshift(fft)
    xx = scipy.fftpack.ifft2(fft)
    y = abs(xx[:, :, np.newaxis])
    return y


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'


### DAGAN VERSION MASK
    mask_name = 'gaussian2d'
    mask_perc = '30'

    if mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join('mask', 'Gaussian2D', "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join('mask', 'Gaussian1D', "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join('mask', 'Poisson2D', "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))


### RefineGAN VERSION MASK
    # mask_name = 'radial'  # cartes, gauss, radial, spiral
    # mask_perc = '30'  # 10...90
    # imask = cv2.imread(os.path.join('mask', mask_name, '{}_{}.tif'.format(mask_name, mask_perc)), 0)/255
    # mask = scipy.fftpack.fftshift(imask)



    H_path = './trainsets/trainH'
    L_path = './trainsets/trainL_G2D30'

    isExists = os.path.exists(L_path)
    if not isExists:
        os.makedirs(os.path.join(L_path))

    main(H_path, L_path, mask)

    H_path = './testsets/testH'
    L_path = './testsets/testL_G2D30'

    isExists = os.path.exists(L_path)
    if not isExists:
        os.makedirs(os.path.join(L_path))

    main(H_path, L_path, mask)



