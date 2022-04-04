
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
        from models.model_swinmr import MRI_NPI as M

    elif model == 'swinmr_stgan':
        from models.model_swinmr_stgan import MRI_STGAN as M

    elif model == 'swinmr_eesgan':
        from models.model_swinmr_eesgan import MRI_EESGAN as M

    elif model == 'swinmr_tesgan':
        from models.model_swinmr_tesgan import MRI_TESGAN as M

    elif model == 'sdaut_npi':
        from models.model_sdaut import MRI_SDAUT as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
