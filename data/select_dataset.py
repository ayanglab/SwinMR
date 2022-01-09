

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['miccai']:
        from data.dataset_MICCAI import DatasetMICCAI as D

    elif dataset_type in ['cc']:
        from data.dataset_CC import DatasetCC as D

    elif dataset_type in ['fastmri']:
        from data.dataset_FastMRI import DatasetCC as D

    elif dataset_type in ['brats17']:
        from data.dataset_BraTS17_npy import DatasetBraTS17 as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
