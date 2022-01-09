## Different Options

`train_swinmr_ori.json`
- single-channel data (MICCAI 2013) 
- only pixel-wise L1 loss


`train_swinmr_npi.json` non parallel image (without Sensitivity Map)
- single-channel data (CC) 
- pixel-wise L1 loss, frequency loss, perceptual loss (VGG)

`train_swinmr_pi.json` parallel image (using Sensitivity Map)
- multi-channel data (CC)  
- pixel-wise L1 loss, frequency loss, perceptual loss (VGG)










`train_swinmr_gan.json`
- multi-channel data (CC) 
- image loss, frequency loss, perceptual loss, adversarial loss
