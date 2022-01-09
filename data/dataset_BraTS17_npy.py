import random
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
from models.select_mask import define_Mask
from math import floor
from skimage.transform import resize

class DatasetBraTS17(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetBraTS17, self).__init__()
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
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_H, 'Error: Raw path is empty.'


        if self.is_mini_dataset:

            index = list(range(0, len(self.paths_H)))
            index_chosen = random.sample(index, round(self.mini_dataset_prec * len(self.paths_H)))
            self.paths_H_new = []
            for i in index_chosen:
                self.paths_H_new.append(self.paths_H[i])
            self.paths_H = self.paths_H_new

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

        H_path = self.paths_H[index]
        file_name = os.path.basename(H_path)[3:-4]
        img_H, _ = load_images(H_path, None, isSM=False)
        img_H_r = resize(img_H, (256, 256, img_H.shape[2]), order=1)
        img_L_r = undersample_kspace(img_H_r, mask, is_noise, noise_level, noise_var)
        img_L = resize(img_L_r, (img_H.shape[0], img_H.shape[1], img_H.shape[2]), order=1)
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
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), \
                                       util.augment_img(patch_H, mode=mode)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(patch_L), \
                                            util.uint2tensor3(patch_H)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'mask': mask, 'name': file_name}

    def __len__(self):
        return len(self.paths_H)
