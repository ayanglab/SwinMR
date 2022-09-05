'''
# -----------------------------------------
Data Loader
CC-SAG-NPI d.1.1
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import random
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
from models.select_mask import define_Mask


class DatasetCCsagnpi(data.Dataset):

    def __init__(self, opt):
        super(DatasetCCsagnpi, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size = self.opt['H_size']
        self.is_noise = self.opt['is_noise']
        self.noise_level = self.opt['noise_level']
        self.noise_var = self.opt['noise_var']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']

        # get data path of image & sensitivity map
        self.paths_raw = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []
        self.paths_SM = []
        for path in self.paths_raw:
            if 'imgGT' in path:
                self.paths_H.append(path)
            elif 'SensitivityMaps' in path:
                self.paths_SM.append(path)
            else:
                raise ValueError('Error: Unknown filename is in raw path')

        if self.is_mini_dataset:
            pass

        # get mask
        self.mask = define_Mask(self.opt)

    def __getitem__(self, index):

        mask = self.mask
        is_noise = self.is_noise
        noise_level = self.noise_level
        noise_var = self.noise_var

        # get gt image
        H_path = self.paths_H[index]
        img_H, _ = self.load_images(H_path, 0, isSM=False)

        # get zf image
        img_L = self.undersample_kspace(img_H, mask, is_noise, noise_level, noise_var)

        # get image information
        image_name_ext = os.path.basename(H_path)
        img_name, ext = os.path.splitext(image_name_ext)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------

            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(patch_L), util.float2tensor3(patch_H)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(img_L), util.float2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'mask': mask, 'SM': _, 'img_info': img_name}

    def __len__(self):
        return len(self.paths_H)

    def load_images(self, H_path, SM_path, isSM=True):
        # load GT
        gt = np.load(H_path).astype(np.float32)

        gt = np.reshape(gt, (gt.shape[0], gt.shape[1], 1))
        # # 0 ~ 1
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        # load SM
        if isSM == True:
            sm = np.load(SM_path).astype(np.float32)[:, :, :, 1]

            # sm = np.reshape(sm[:, :, :, 1], (256, 256, 12))

            # 0 ~ 1
            sm = (sm - sm.min()) / (sm.max() - sm.min())

            return gt, sm
        else:
            return gt, 0

    def undersample_kspace(self, x, mask, is_noise, noise_level, noise_var):

        fft = fft2(x[:, :, 0])
        fft = fftshift(fft)
        fft = fft * mask
        if is_noise:
            fft = fft + self.generate_gaussian_noise(fft, noise_level, noise_var)
        fft = ifftshift(fft)
        xx = ifft2(fft)
        xx = np.abs(xx)

        x = xx[:, :, np.newaxis]

        return x

    def generate_gaussian_noise(self, x, noise_level, noise_var):
        spower = np.sum(x ** 2) / x.size
        npower = noise_level / (1 - noise_level) * spower
        noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
        return noise