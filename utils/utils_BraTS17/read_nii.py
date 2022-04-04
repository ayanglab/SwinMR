from utils.utils_BraTS17.data_loader import *
from utils.parse_config import parse_config
import cv2

preserving_ratio = 0.5

config = parse_config('./config17/dataset.txt')
config_data = config['data']

dataloader = DataLoader(config_data)
dataloader.load_data()

image_num = dataloader.get_total_image_number()

images_flair = []
images_t1 = []
images_t1ce = []
images_t2 = []

# image_num: the number of patients (3D volumes)
for idx in range(image_num):
    # data: data with noise, weight: a 0/1 mask for useful area, in_size: size of volumes
    raw_data = dataloader.get_image_data_with_name(idx)
    [data, weight, patient_names, image_names, bbox, in_size] = raw_data

    # data: a list of different model [flair, t1, t1ce, t2]
    for idx_type in range(len(data)):

        # get volumes
        volumes = data[idx_type]

        # axial
        [D, H, W] = volumes.shape
        volumes_01 = (volumes - volumes.min()) / (volumes.max() - volumes.min())
        volumes_01 = volumes_01 * weight

        for idx_slice in range(volumes.shape[0]):
            # non-zero > ratio
            if float(np.count_nonzero(weight[idx_slice, :, :])) / weight[idx_slice, :, :].size >= preserving_ratio:
                image = volumes_01[idx_slice, :, :] * 255
                if idx_type == 0:
                    images_flair.append(image)
                elif idx_type == 1:
                    images_t1.append(image)
                elif idx_type == 2:
                    images_t1ce.append(image)
                elif idx_type == 3:
                    images_t2.append(image)

# make dir
for type_name in ['flair', 't1', 't1ce', 't2']:
    save_path = os.path.join('/media/NAS01/BraTS17', 'trainsets', type_name)
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)

    if type_name is 'flair':
        for i in range(len(images_flair)):
            cv2.imwrite(os.path.join(save_path, 'GT_{:05d}.png'.format(i)), images_flair[i])
    elif type_name is 't1':
        for i in range(len(images_flair)):
            cv2.imwrite(os.path.join(save_path, 'GT_{:05d}.png'.format(i)), images_t1[i])
    elif type_name is 't1ce':
        for i in range(len(images_flair)):
            cv2.imwrite(os.path.join(save_path, 'GT_{:05d}.png'.format(i)), images_t1ce[i])
    elif type_name is 't2':
        for i in range(len(images_flair)):
            cv2.imwrite(os.path.join(save_path, 'GT_{:05d}.png'.format(i)), images_t2[i])
