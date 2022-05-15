


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['miccai2013']:
        from data.dataset_MICCAI2013 import DatasetMICCAI2013 as D

    elif dataset_type in ['ccsagnpi']:
        from data.dataset_CCsagnpi import DatasetCCsagnpi as D

    elif dataset_type in ['ccsagpi']:
        from data.dataset_CCsagpi import DatasetCCsagpi as D

    elif dataset_type in ['ccaxinpi']:
        from data.dataset_CCaxinpi import DatasetCCaxinpi as D

    elif dataset_type in ['fastmri']:
        from data.dataset_FastMRI import DatasetFastMRI as D

    elif dataset_type in ['brats17']:
        from data.dataset_BraTS17 import DatasetBraTS17 as D

    elif dataset_type in ['brats17_256']:
        from data.dataset_BraTS17_256 import DatasetBraTS17 as D

    elif dataset_type in ['kaggle']:
        from data.dataset_Kaggle import DatasetKaggle as D

    elif dataset_type in ['kaggle_automask']:
        from data.dataset_KaggleAutomask import DatasetKaggleAutomask as D

    elif dataset_type in ['lascarqs2022']:
        from data.dataset_LAScarQS2022 import DatasetLAScarQS2022 as D

    elif dataset_type in ['lascarqs2022_224']:
        from data.dataset_LAScarQS2022_224 import DatasetLAScarQS2022 as D

    elif dataset_type in ['ilsvrc2012']:
        from data.dataset_ILSVRC2012 import DatasetILSVRC2012 as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
