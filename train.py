#! /usr/bin/python
# -*- coding: utf8 -*-


# ------------- import python necessary libraries ------------------------------
import os  
import time
import pandas as pd
import numpy as np
import scipy, multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import training
import tensorlayer as tl
# ---------------- import local libraries---------------------------------------
from model import get_G, get_D
from config import config 
from open_data import open_data 

MISSING = object()

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
n_epoch_disc = config.TRAIN.n_epochDicsriminator
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
name = config.Name

# create folders to save result images and trained models
save_dir = "/content/drive/MyDrive/Colab Notebooks/ASRGAN/samples"
save_dir_losses = "/content/drive/MyDrive/Colab Notebooks/ASRGAN/losses/"
tl.files.exists_or_mkdir(save_dir)

checkpoint_dir = "/content/drive/MyDrive/Colab Notebooks/ASRGAN/models"
tl.files.exists_or_mkdir(checkpoint_dir)

def get_train_data():
    if train_ASRGAN or train_SRGAN: 
        # load dataset
        train_hr_imgs = open_data(config.TRAIN.hr_img_path)  
        train_lr_imgs = open_data(config.TRAIN.lr_img_path) 
        
        # Convert pickle file to tensor
        train_hr_imgs = tf.convert_to_tensor(train_hr_imgs)
        train_lr_imgs = tf.convert_to_tensor(train_lr_imgs)      
     
    # dataset API
    def generator_train():
        for img_hr,img_lr in zip(train_hr_imgs,train_lr_imgs):
            img_lr = tf.image.resize(img_lr, size=[192, 192])
            yield tf.stack([img_hr,img_lr], axis=0)

    def _map_fn_train(img):
        patch = tf.image.random_crop(img, [2, 192, 192, 1])
        hr_patch = patch[0]
        lr_patch = tf.image.resize(patch[1], size=[48, 48])
        return lr_patch, hr_patch
    
    train_ds = tf.data.Dataset.from_generator(generator_train,
                                              output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)    
    return train_ds

#############----- train the SR model based on adapted-SRGAN (ASRGAN)- --########
################################################################################

def train_ASRGAN (g= '', d= '' , vgg= '', name = ''):  
    #prepare models
    G = get_G((batch_size, 48, 48, 1))
    D = get_D((batch_size, 192, 192, 1))
    VGG = tl.models.vgg19(pretrained=False, end_with='pool4', mode='static') #  It is not possible to download pre-train vgg19 weights directly in colab. So, load the vgg19 weights manually.

    #load model weights
    if g is not MISSING:
        G.load_weights(g)
    if d is not MISSING:
        D.load_weights(d)
    if vgg is not MISSING:
      VGG.load_weights(vgg)
    
    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data()
    
    # This part is optional : prepare the test dataset. Usually, GPU is running out of memory.
    test_ds_hr = open_data(config.TRAIN.hr_img_path)
    test_ds_lr = open_data(config.TRAIN.lr_img_path)
    Y = np.random.randint(config.TRAIN.batch_size, size = test_ds_hr.shape[0])
    test_ds_hr = open_data(config.TRAIN.hr_img_path)[Y]
    test_ds_lr = open_data(config.TRAIN.lr_img_path)[Y]
    
    # initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)

    #---------------------------   Initializater--------------------------------
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size:                                # If the remaining data in this step < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)                    #G.trainable_weights
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))       #G.trainable_weights
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
            loss_file = open(save_dir_losses +'loss_init.txt' , 'a+')
            loss_file.write('epoch%d : mse_loss = %s \n' %(epoch, mse_loss))
            loss_file.close() 


        if (epoch != 0) and (epoch % 100 == 0):   
            #save the images
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_init_{}.tif'.format(epoch)))
        

    # ------------  adversarial learning (G, D) --------------------------------
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        
        #  -----------------------  shuffle dataset ----------------------------
        train_ds.shuffle(buffer_size = 1000)
        
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size:                                # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)                                      # Generate fake high resolution image 
                logits_fake = D(fake_patchs)                                    # Pass generated fake high resolution image from discriminator 
                
                # The discriminator and the generator are not training at the same epoch. 
                # First, the discriminator must be trained.Then, make the discriminator non-trainabl and training the generator.
                
                # -------------------  Generator -------------------------------
                if epoch >= n_epoch_disc:
                  feature_fake = VGG((fake_patchs+1)/2.)                         # the pre-trained VGG uses the input range of [0, 1]
                  feature_real = VGG((hr_patchs+1)/2.)
                  g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))   # ----> generator loss
                  mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                  vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                  g_loss = mse_loss + vgg_loss + g_gan_loss                      # ------- > perceptual loss
                  grad = tape.gradient(g_loss, G.trainable_weights)              #G.trainable_weights
                  g_optimizer.apply_gradients(zip(grad, G.trainable_weights))    #G.trainable_weights

                # ------------------ Discriminator ----------------------------- 
                else:
                  logits_real = D(hr_patchs)
                  d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                  d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                  d_loss = d_loss1 + d_loss2
                  grad = tape.gradient(d_loss, D.trainable_weights)               #D.trainable_weights
                  d_optimizer.apply_gradients(zip(grad, D.trainable_weights))     #D.trainable_weights  
                  
                  print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}) d_loss: {:.3f}".format(
                  epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, d_loss))

                  loss_file = open(save_dir_losses +'loss_discriminator.txt' , 'a+')
                  loss_file.write('epoch%d : mse_loss = %s; discriminator_loss = %f\n' %(epoch, mse_loss, d_loss ))
                  loss_file.close() 

            if epoch >= n_epoch_disc:
              print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.6f}) d_loss: {:.6f}".format(
              (epoch-n_epoch_disc), n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

              loss_file = open(save_dir_losses +'loss_generator.txt' , 'a+')
              loss_file.write('epoch%d : gan_loss = %s; mse_loss = %s; vgg_loss = %s ; perc_loss = %s; discriminator_loss = %f\n' %((epoch-n_epoch_disc),g_gan_loss, \
              mse_loss,vgg_loss, g_loss,d_loss ))
              loss_file.close()           
            
        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch-n_epoch_disc != 0) and ((epoch-n_epoch_disc) % 100 == 0):  
            
            #save reconstruction and weights
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'ASRGAN_train_g_{}.tif'.format(epoch-n_epoch_disc)))
            G.save_weights(os.path.join(checkpoint_dir, 'ASRGAN_g_{}.h5'.format(epoch-n_epoch_disc)))
            D.save_weights(os.path.join(checkpoint_dir, 'ASRGAN_d_{}.h5'.format(epoch-n_epoch_disc)))
            VGG.save_weights(os.path.join(checkpoint_dir, 'ASRGAN_vgg_{}.h5'.format(epoch-n_epoch_disc)))

#################----- train the SR model based on conventinal SRGAN --#########
################################################################################

def train_SRGAN (g='', d='', vgg = '', name = ''):

    G = get_G((batch_size, 48, 48, 1))
    D = get_D((batch_size, 192, 192, 1))
    VGG = tl.models.vgg19(pretrained=False, end_with='pool4', mode='static') # Cannot download pre-train vgg19 weights directly


    #load model weights
    if g is not MISSING:
        G.load_weights(g)
    if d is not MISSING:
        D.load_weights(d)
    if vgg is not MISSING:
      VGG.load_weights(vgg)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data()
    
    
    #préparer le jeu de données de test
    test_ds_hr = open_data(config.TRAIN.hr_img_path)
    test_ds_lr = open_data(config.TRAIN.lr_img_path)
    Y = np.random.randint(config.TRAIN.batch_size, size = test_ds_hr.shape[0])
    test_ds_hr = open_data(config.TRAIN.hr_img_path)[Y]
    test_ds_lr = open_data(config.TRAIN.lr_img_path)[Y]
    

    ## initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size) 
  
    
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: 
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))

        if (epoch != 0) and (epoch % 100 == 0):
        
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_init_{}.tif'.format(epoch)))
    
    train_mse, train_vgg, train_adv, test_mse, test_vgg, test_adv = [], [], [], [], [], []
    
    ## adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        
        #mélanger le jeu de données
        train_ds.shuffle(buffer_size = 1000)        
  
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):

            if lr_patchs.shape[0] != batch_size: 
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:

                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)

                feature_fake = VGG((fake_patchs+1)/2.) 
                feature_real = VGG((hr_patchs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-4 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))  # Adverseroal loss
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

            loss_file = open(save_dir_losses +'SRGAN_losses.txt' , 'a+')
            loss_file.write('epoch%d : gan_loss = %s; mse_loss = %s; vgg_loss = %s ; perc_loss = %s; discriminator_loss = %f\n' %(epoch,g_gan_loss, \
            mse_loss,vgg_loss, g_loss,d_loss ))
            loss_file.close()           

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)   
            lr_v.assign(lr_init * new_lr_decay)             
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 100 == 0):  
            
            # save reconstruction and weights
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'SRGAN_train_g_{}.tif'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'SRGAN_g_{}.h5'.format(epoch)))
            D.save_weights(os.path.join(checkpoint_dir, 'SRGAN_d_{}.h5'.format(epoch)))
            VGG.save_weights(os.path.join(checkpoint_dir, 'SRGAN_vgg_{}.h5'.format(epoch)))



#%%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='ASRGAN', help='ASRGAN, SRGAN')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode


    g = MISSING
    d = MISSING
    vgg = '/content/drive/MyDrive/Colab Notebooks/ASRGAN/models/vgg.h5'

    if tl.global_flag['mode'] == 'ASRGAN':
        train_ASRGAN (g=g, d=d, vgg = vgg, name = name)
    elif tl.global_flag['mode'] == 'SRGAN':
        train_SRGAN (g=g, d=d, vgg = vgg, name = name)
    else:
        raise Exception("Unknow --mode")
        

