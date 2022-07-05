


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['ccsagnpi']:
        from data.dataset_CCsagnpi import DatasetCCsagnpi as D

    elif dataset_type in ['ccsagpi']:
        from data.dataset_CCsagpi import DatasetCCsagpi as D

    elif dataset_type in ['brats17']:
        from data.dataset_BraTS17 import DatasetBraTS17 as D

    elif dataset_type in ['brats17_256']:
        from data.dataset_BraTS17_256 import DatasetBraTS17 as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
