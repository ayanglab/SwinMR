import random
import torch.utils.data as data
import utils.utils_image as util
import utils.utils_swinmr as utils
from models.select_mask import define_Mask
from math import floor


class FastMRI(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(FastMRI, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 1
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 128

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


        # ------------------------------------
        # get mask
        # ------------------------------------

        self.mask = define_Mask(self.opt)

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------

        H_path = self.paths_H[floor(index/2)]
        SM_path = self.paths_SM[floor(index/2)]

        img_H, Sensitivity_Map = utils.load_images(H_path, SM_path, 47, 10, isSM=True)

        img_L = utils.undersample_kspace(img_H, self.mask)

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

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'mask': self.mask, 'SM': Sensitivity_Map}

    def __len__(self):
        return len(self.paths_raw)
