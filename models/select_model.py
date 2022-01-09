
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']

    if model == 'swinmr_pi':
        from models.model_swinmr_pi import MRI_PI as M

    elif model == 'swinmr_npi':
        from models.model_swinmr_npi import MRI_NPI as M

    elif model == 'swinmr_ori':
        from models.model_swinmr_ori import MRI_Ori as M

    elif model == 'swinmr_gan':
        from models.model_gan import MRI_GAN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
