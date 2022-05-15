import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_early_stopping

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tensorboardX import SummaryWriter
import time



def main(json_path=''):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],  net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # tensorbordX log
    logger_tensorboard = SummaryWriter(os.path.join(opt['path']['log']))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    early_stopping = utils_early_stopping.EarlyStopping(patience=opt['train']['early_stopping_num'])
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(100000000):  # keep running

        for i, train_data in enumerate(train_loader):

            current_step += 1
            # if current_step == 1:
            #     time1 = time.time()
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # if current_step == 100:
            #     time2 = time.time()
            #     time_used_100_steps = time2 - time1
            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                # message += 'Time Used: {:.8e} '.format(time_used_100_steps)
                logger.info(message)

                # record train loss
                logger_tensorboard.add_scalar('Learning Rate', model.current_learning_rate(), global_step=current_step)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss', logs['G_loss'], global_step=current_step)

                if 'G_loss_image' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_image', logs['G_loss_image'],
                                                  global_step=current_step)
                if 'G_loss_frequency' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_frequency', logs['G_loss_frequency'],
                                                  global_step=current_step)
                if 'G_loss_preceptual' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_preceptual', logs['G_loss_preceptual'],
                                                  global_step=current_step)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0

                for idx, test_data in enumerate(test_loader):
                    image_name_ext = os.path.basename(test_data['H_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()

                    L_img = util.tensor2uint(visuals['L'])
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------

                    if idx < 10:
                        save_img_path = os.path.join(img_dir, 'ZF_{:5d}.png'.format(current_step))
                        util.imsave(L_img, save_img_path)
                        save_img_path = os.path.join(img_dir, 'Recon_{:5d}.png'.format(current_step))
                        util.imsave(E_img, save_img_path)
                        save_img_path = os.path.join(img_dir, 'GT_{:5d}.png'.format(current_step))
                        util.imsave(H_img, save_img_path)

                    # -----------------------
                    # calculate PSNR SSIM
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | PSNR: {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                    logger.info('{:->4d}--> {:>10s} | SSIM: {:<4.4f}'.format(idx, image_name_ext, current_ssim))

                    avg_psnr = current_psnr + avg_psnr
                    avg_ssim = current_ssim + avg_ssim

                    if idx > 10:
                        break

                avg_psnr = avg_psnr / (idx + 1)
                avg_ssim = avg_ssim / (idx + 1)

                # testing log
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average SSIM : {:<.4f}dB\n'.format(epoch, current_step, avg_ssim))

                logger_tensorboard.add_scalar('VALIDATION PSNR', avg_psnr, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION SSIM', avg_ssim, global_step=current_step)

                # early stopping
                if opt['train']['is_early_stopping']:
                    early_stopping(avg_psnr, model, epoch, current_step)
                    if early_stopping.is_save:
                        logger.info('Saving the model by early stopping')
                        model.save(f'best_{current_step}')
                    if early_stopping.early_stop:
                        print("Early stopping!")
                        break

    print("Training Stop")


if __name__ == '__main__':

    # Train SwinMR CCpi
    main('options/SwinMR/train_swinmr_CCpi_G1D30.json')
    # Train SwinMR CCnpi
    main('options/SwinMR/train_swinmr_CCnpi_G1D30.json')
