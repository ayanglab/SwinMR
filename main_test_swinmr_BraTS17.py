import argparse
import cv2
import csv
import sys
import numpy as np
from collections import OrderedDict
import os
import torch
from utils import utils_option as option
from torch.utils.data import DataLoader
from models.network_swinir import SwinIR as net
from utils import utils_image as util
from utils.utils_swinmr import sobel
from data.select_dataset import define_Dataset


def main(json_path='options/test_swinmr_pi.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(opt['model_path']):
        print(f"loading model from {opt['model_path']}")
    else:
        print('can\'t find model.')

    model = define_model(opt)
    model.eval()
    model = model.to(device)

    # setup folder and path
    save_dir, border, window_size = setup(opt)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_e'] = []
    test_results['ssim_e'] = []
    test_results['zf_psnr'] = []
    test_results['zf_ssim'] = []
    test_results['zf_psnr_e'] = []
    test_results['zf_ssim_e'] = []


    with open(os.path.join(save_dir, 'results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR', 'Edge_SSIM', 'Edge_PSNR'])

    with open(os.path.join(save_dir, 'results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'Edge_SSIM', 'Edge_SSIM_STD',
                         'Edge_PSNR', 'Edge_PSNR_STD',
                         'FID'])

    with open(os.path.join(save_dir, 'zf_results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR', 'Edge_SSIM', 'Edge_PSNR'])

    with open(os.path.join(save_dir, 'zf_results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'Edge_SSIM', 'Edge_SSIM_STD',
                         'Edge_PSNR', 'Edge_PSNR_STD',
                         'FID'])

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    dataset_opt = opt['datasets']['test']

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    data_mask = []
    for idx, test_data in enumerate(test_loader):


        img_gt = test_data['H'].to(device)
        img_lq = test_data['L'].to(device)
        name = test_data['name'][0]
        
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old

            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            img_gt = torch.cat([img_gt, torch.flip(img_gt, [2])], 2)[:, :, :h_old + h_pad, :]
            img_gt = torch.cat([img_gt, torch.flip(img_gt, [3])], 3)[:, :, :, :w_old + w_pad]

            img_gen = model(img_lq)

            img_lq = img_lq[..., :h_old * opt['scale'], :w_old * opt['scale']]
            img_gt = img_gt[..., :h_old * opt['scale'], :w_old * opt['scale']]
            img_gen = img_gen[..., :h_old * opt['scale'], :w_old * opt['scale']]

            edge_lq = sobel(img_lq, device)
            edge_gt = sobel(img_gt, device)
            edge_gen = sobel(img_gen, device)

            diff_gen_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_gen)), 10)
            diff_lq_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_lq)), 10)
            diff_edge_gen_x10 = torch.mul(torch.abs(torch.sub(edge_gt, edge_gen)), 10)
            diff_edge_lq_x10 = torch.mul(torch.abs(torch.sub(edge_gt, edge_lq)), 10)

        # make dir of npy
        isExists = os.path.exists(os.path.join(save_dir, 'npy', 'ZF'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'npy', 'ZF'))
        isExists = os.path.exists(os.path.join(save_dir, 'npy', 'GT'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'npy', 'GT'))
        isExists = os.path.exists(os.path.join(save_dir, 'npy', 'Recon'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'npy', 'Recon'))


        # make dir of png
        isExists = os.path.exists(os.path.join(save_dir, 'png', 'ZF'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'ZF'))
        isExists = os.path.exists(os.path.join(save_dir, 'png', 'GT'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'GT'))
        isExists = os.path.exists(os.path.join(save_dir, 'png', 'Recon'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'Recon'))

        isExists = os.path.exists(os.path.join(save_dir, 'png', 'Edge_ZF'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'Edge_ZF'))
        isExists = os.path.exists(os.path.join(save_dir, 'png', 'Edge_GT'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'Edge_GT'))
        isExists = os.path.exists(os.path.join(save_dir, 'png', 'Edge_Recon'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'Edge_Recon'))
        isExists = os.path.exists(os.path.join(save_dir, 'png', 'Different'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'png', 'Different'))

        # 0~1
        img_lq = img_lq.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_gt = img_gt.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        img_gen = img_gen.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        edge_lq = edge_lq.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        edge_gt = edge_gt.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        edge_gen = edge_gen.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_edge_gen_x10 = diff_edge_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_edge_lq_x10 = diff_edge_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        # save npy
        np.save(os.path.join(save_dir, 'npy', 'ZF', f'zf_{name}.npy'), img_lq)
        np.save(os.path.join(save_dir, 'npy', 'GT', f'gt_{name}.npy'), img_gt)
        np.save(os.path.join(save_dir, 'npy', 'Recon', f'rc_{name}.npy'), img_gen)

        #
        preserving_ratio = 0.3
        t = np.clip((img_gt - 0.01), 0.0, 1.0)
        A = float(np.count_nonzero(t)) / t.size >= preserving_ratio
        B = np.isnan(t).any()
        if A and (not B):
            isValid = True
            data_mask.append(1)
        else:
            isValid = False
            data_mask.append(0)

        # 0~255
        img_lq = (img_lq * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gen = (img_gen * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        edge_lq = (edge_lq * 255.0).round().astype(np.uint8)  # float32 to uint8
        edge_gt = (edge_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
        edge_gen = (edge_gen * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_edge_gen_x10 = (diff_edge_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_edge_lq_x10 = (diff_edge_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8

        # save png
        cv2.imwrite(os.path.join(save_dir, 'png', 'ZF', f'zf_{name}.png'), img_lq)
        cv2.imwrite(os.path.join(save_dir, 'png', 'GT', f'gt_{name}.png'), img_gt)
        cv2.imwrite(os.path.join(save_dir, 'png', 'Recon', f'rc_{name}.png'), img_gen)

        # cv2.imwrite(os.path.join(save_dir, 'png', 'Edge_ZF', f'edge_zf_{name}.png'), edge_lq)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Edge_GT', f'edge_gt_{name}.png'), edge_gt)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Edge_Recon', f'edge_rc_{name}.png'), edge_gen)
        #
        # diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
        # diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)
        # diff_edge_gen_x10_color = cv2.applyColorMap(diff_edge_gen_x10, cv2.COLORMAP_JET)
        # diff_edge_lq_x10_color = cv2.applyColorMap(diff_edge_lq_x10, cv2.COLORMAP_JET)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_rc_{name}.png'), diff_gen_x10_color)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_zf_{name}.png'), diff_lq_x10_color)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_edge_rc_{name}.png'), diff_edge_gen_x10_color)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_edge_zf_{name}.png'), diff_edge_lq_x10_color)

        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_rc_{name}.png'), diff_gen_x10)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_zf_{name}.png'), diff_lq_x10)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_edge_rc_{name}.png'), diff_edge_gen_x10)
        # cv2.imwrite(os.path.join(save_dir, 'png', 'Different', f'Diff_edge_zf_{name}.png'), diff_edge_lq_x10)

        # evaluate psnr/ssim
        psnr = util.calculate_psnr_single(img_gen, img_gt, border=border)
        ssim = util.calculate_ssim_single(img_gen, img_gt, border=border)
        psnr_e = util.calculate_psnr_single(edge_gen, edge_gt, border=border)
        ssim_e = util.calculate_ssim_single(edge_gen, edge_gt, border=border)

        if isValid:
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            print('Testing {:d} - PSNR: {:.2f} dB; SSIM: {:.4f}; '.format(idx, psnr, ssim))
            test_results['psnr_e'].append(psnr_e)
            test_results['ssim_e'].append(ssim_e)
            print('Testing {:d} Edge - PSNR: {:.2f} dB; SSIM: {:.4f}; '.format(idx, psnr_e, ssim_e))

            with open(os.path.join(save_dir, 'results.csv'), 'a') as cf:
                writer = csv.writer(cf)
                writer.writerow(['SwinMR', dataset_opt['mask'],
                                 test_results['ssim'][idx], test_results['psnr'][idx],
                                 test_results['ssim_e'][idx], test_results['psnr_e'][idx]])
        else:
            test_results['psnr'].append(np.nan)
            test_results['ssim'].append(np.nan)
            print('Testing {:d} - Invalid Data;'.format(idx))
            test_results['psnr_e'].append(np.nan)
            test_results['ssim_e'].append(np.nan)
            print('Testing {:d} Edge - Invalid Data;'.format(idx))

        # evaluate psnr/ssim zf
        zf_psnr = util.calculate_psnr_single(img_gt, img_lq, border=border)
        zf_ssim = util.calculate_ssim_single(img_gt, img_lq, border=border)
        zf_psnr_e = util.calculate_psnr_single(edge_gt, edge_lq, border=border)
        zf_ssim_e = util.calculate_ssim_single(edge_gt, edge_lq, border=border)

        if isValid:
            test_results['zf_psnr'].append(zf_psnr)
            test_results['zf_ssim'].append(zf_ssim)
            print('ZF Testing {:d} - PSNR: {:.2f} dB; SSIM: {:.4f}; '.format(idx, zf_psnr, zf_ssim))
            test_results['zf_psnr_e'].append(zf_psnr_e)
            test_results['zf_ssim_e'].append(zf_ssim_e)
            print('ZF Testing {:d} Edge - PSNR: {:.2f} dB; SSIM: {:.4f}; '.format(idx, zf_psnr_e, zf_ssim_e))

            with open(os.path.join(save_dir, 'zf_results.csv'), 'a') as cf:
                writer = csv.writer(cf)
                writer.writerow(['ZF', dataset_opt['mask'],
                                 test_results['zf_ssim'][idx], test_results['zf_psnr'][idx],
                                 test_results['zf_ssim_e'][idx], test_results['zf_psnr_e'][idx]])
        else:
            test_results['zf_psnr'].append(np.nan)
            test_results['zf_ssim'].append(np.nan)
            print('ZF Testing {:d} - Invalid Data;'.format(idx))
            test_results['zf_psnr_e'].append(np.nan)
            test_results['zf_ssim_e'].append(np.nan)
            print('ZF Testing {:d} Edge - Invalid Data;'.format(idx))

    # summarize psnr/ssim
    ave_psnr = np.nanmean(test_results['psnr'])
    std_psnr = np.nanstd(test_results['psnr'], ddof=1)
    ave_ssim = np.nanmean(test_results['ssim'])
    std_ssim = np.nanstd(test_results['ssim'], ddof=1)

    ave_psnr_e = np.nanmean(test_results['psnr_e'])
    std_psnr_e = np.nanstd(test_results['psnr_e'], ddof=1)
    ave_ssim_e = np.nanmean(test_results['ssim_e'])
    std_ssim_e = np.nanstd(test_results['ssim_e'], ddof=1)

    print('\n{} \n-- Average PSNR {:.2f} dB ({:.4f} dB)\n-- Average SSIM  {:.4f} ({:.6f})'
          .format(save_dir, ave_psnr, std_psnr, ave_ssim, std_ssim))
    print('\n{} \n-- Edge Average PSNR {:.2f} dB ({:.4f} dB)\n-- Edge Average SSIM  {:.4f} ({:.6f})'
          .format(save_dir, ave_psnr_e, std_psnr_e, ave_ssim_e, std_ssim_e))

    # summarize psnr/ssim zf
    zf_ave_psnr = np.nanmean(test_results['zf_psnr'])
    zf_std_psnr = np.nanstd(test_results['zf_psnr'], ddof=1)
    zf_ave_ssim = np.nanmean(test_results['zf_ssim'])
    zf_std_ssim = np.nanstd(test_results['zf_ssim'], ddof=1)

    zf_ave_psnr_e = np.nanmean(test_results['zf_psnr_e'])
    zf_std_psnr_e = np.nanstd(test_results['zf_psnr_e'], ddof=1)
    zf_ave_ssim_e = np.nanmean(test_results['zf_ssim_e'])
    zf_std_ssim_e = np.nanstd(test_results['zf_ssim_e'], ddof=1)

    print('\n{} \n-- ZF Average PSNR {:.2f} dB ({:.4f} dB)\n-- ZF Average SSIM  {:.4f} ({:.6f})'
          .format(save_dir, zf_ave_psnr, zf_std_psnr, zf_ave_ssim, zf_std_ssim))
    print('\n{} \n-- ZF Edge Average PSNR {:.2f} dB ({:.4f} dB)\n-- ZF Edge Average SSIM  {:.4f} ({:.6f})'
          .format(save_dir, zf_ave_psnr_e, zf_std_psnr_e, zf_ave_ssim_e, zf_std_ssim_e))

    # FID
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'png', 'GT'),
        os.path.join(save_dir, 'png', 'Recon'))).read()
    print(log)
    fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['SwinMR', dataset_opt['mask'],
                         ave_ssim, std_ssim,
                         ave_psnr, std_psnr,
                         ave_ssim_e, std_ssim_e,
                         ave_psnr_e, std_psnr_e,
                         fid])
    # FID ZF
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'png', 'ZF'),
        os.path.join(save_dir, 'png', 'Recon'))).read()
    print(log)
    zf_fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'zf_results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['ZF', dataset_opt['mask'],
                         zf_ave_ssim, zf_std_ssim,
                         zf_ave_psnr, zf_std_psnr,
                         zf_ave_ssim_e, zf_std_ssim_e,
                         zf_ave_psnr_e, zf_std_psnr_e,
                         zf_fid])

def define_model(args):

    model = net(upscale=1, in_chans=1, img_size=256, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=args['netG']['embed_dim'], num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = torch.load(args['model_path'])
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model


def setup(args):

    save_dir = f"results/{args['task']}/{args['model_name']}"
    border = 0
    window_size = 8

    return save_dir, border, window_size


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    main('options/test/test_swinmr_brats17.json')

    # main('options/test/seg_brats17/test_swinmr_flair_G1D30.json')
    # main('options/test/seg_brats17/test_swinmr_t1_G1D30.json')
    # main('options/test/seg_brats17/test_swinmr_t1ce_G1D30.json')
    # main('options/test/seg_brats17/test_swinmr_t2_G1D30.json')
    # main('options/test/seg_brats17/test_swinmr_flair_G1D10.json')
    # main('options/test/seg_brats17/test_swinmr_t1_G1D10.json')
    # main('options/test/seg_brats17/test_swinmr_t1ce_G1D10.json')
    # main('options/test/seg_brats17/test_swinmr_t2_G1D10.json')
