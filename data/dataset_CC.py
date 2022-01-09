import random
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
from models.select_mask import define_Mask
from math import floor


class DatasetCC(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetCC, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size = self.opt['H_size']
        self.is_noise = self.opt['is_noise']
        self.noise_level = self.opt['noise_level']
        self.noise_var = self.opt['noise_var']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']
        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
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
                assert 0, 'Error: Unknown filename is in raw path'

        if self.is_mini_dataset:

            index = list(range(0, len(self.paths_H)))
            index_chosen = random.sample(index, round(self.mini_dataset_prec * len(self.paths_H)))
            self.paths_H_new = []
            self.paths_SM_new = []
            for i in index_chosen:
                self.paths_H_new.append(self.paths_H[i])
                self.paths_SM_new.append(self.paths_SM[i])
            self.paths_H = self.paths_H_new
            self.paths_SM = self.paths_SM_new
        # ------------------------------------
        # get mask
        # ------------------------------------

        self.mask = define_Mask(self.opt)

    def __getitem__(self, index):

        mask = self.mask
        is_noise = self.is_noise
        noise_level = self.noise_level
        noise_var = self.noise_var

        # ------------------------------------
        # get H image
        # ------------------------------------

        H_path = self.paths_H[floor(index/2)]
        SM_path = self.paths_SM[floor(index/2)]

        img_H, Sensitivity_Map = load_images(H_path, SM_path, isSM=True)

        img_L = undersample_kspace(img_H, mask, is_noise, noise_level, noise_var)

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
            patch_SM = Sensitivity_Map[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------

            mode = random.randint(0, 7)
            patch_L, patch_H, patch_SM= util.augment_img(patch_L, mode=mode), \
                                       util.augment_img(patch_H, mode=mode), \
                                       util.augment_img(patch_SM, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H, Sensitivity_Map = util.uint2tensor3(patch_L), \
                                            util.uint2tensor3(patch_H), \
                                            util.uint2tensor3(patch_SM)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'mask': mask, 'SM': Sensitivity_Map}

    def __len__(self):
        return len(self.paths_H)
