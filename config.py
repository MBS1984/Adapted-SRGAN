from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam optimizer
config.TRAIN.batch_size = 8 

config.TRAIN.lr_init = 1e-4  # lr for initializing the generator
# config.TRAIN.lr_init_g = 1e-4
# config.TRAIN.lr_init_d = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 50 
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## Adversarial learning
config.TRAIN.n_epochDicsriminator = 20
config.TRAIN.n_epoch = 1021 
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 1)   

## trainin set path

config.TRAIN.lr_img_path = '/content/drive/MyDrive/Colab Notebooks/ASRGAN/datasets/trainLr.pkl'
config.TRAIN.hr_img_path = '/content/drive/MyDrive/Colab Notebooks/ASRGAN/datasets/trainHr.pkl'

## VGG19 weights path
config.TRAIN.vgg_weights = '/content/drive/MyDrive/Colab Notebooks/ASRGAN/models/vgg.h5'


config.Name = 'ASRGAN'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
