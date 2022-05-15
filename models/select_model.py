
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']

    if model == 'swinmr_pi':
        from models.model_swinmr_pi import MRI_SwinMR_PI as M

    elif model == 'swinmr_npi':
        from models.model_swinmr import MRI_SwinMR_NPI as M

    elif model == 'swinmr_stgan':
        from models.model_swinmr_stgan import MRI_STGAN as M

    elif model == 'swinmr_eesgan':
        from models.model_swinmr_eesgan import MRI_EESGAN as M

    elif model == 'swinmr_tesgan':
        from models.model_swinmr_tesgan import MRI_TESGAN as M

    elif model == 'sdaut_npi':
        from models.model_sdaut import MRI_SDAUT as M

    elif model == 'sdaut_v2s':
        from models.model_sdaut_v2s import MRI_SDAUTv2S as M

    elif model == 'sdaut_v2c':
        from models.model_sdaut_v2c import MRI_SDAUTv2C as M

    elif model == 'unets':
        from models.model_unet_s import MRI_UNetS as M

    elif model == 'swinunets':
        from models.model_swinunets import MRI_SwinUNetS as M

    elif model == 'swinunetr':
        from models.model_swinunetr import MRI_SwinUNetR as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
