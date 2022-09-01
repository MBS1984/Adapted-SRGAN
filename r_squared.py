import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib
import cv2
import scipy
from itertools import chain
from sklearn.metrics import r2_score
from skimage import feature
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from open_data import open_data
from model import get_G
from visualize_utils import cm_to_inch

def r_squared(validLr, validHr, G):

  images_sr  = []

  for i in range(115):
      sr = G(validLr[i].reshape(1, validLr.shape[1], validLr.shape[2], 1)).numpy()
      images_sr.append(sr)   
      
  validLr_resample = []  
  for i in range(115):
      temp = scipy.ndimage.zoom(validLr[i,:,:,0:1], 4, order=0)
      validLr_resample.append(temp)
      
  validLr_resample = np.array(validLr_resample) 
  validLr_resample = validLr_resample [:,:,:,0:1] 
  validLr_resample = (validLr_resample *2.) - 1. 

  img_num = 7 # select a tile number between 0 and 114

  imHr = validHr[img_num,:,:,0]
  imLr = validLr_resample[img_num,:,:,0]
  imSr = images_sr[img_num]
  imSr = imSr[0,:,:,0]

  imHr_Flat =  np.array(list(chain.from_iterable(imHr)))
  imLr_Flat =  np.array(list(chain.from_iterable(imLr)))
  imSr_Flat =  np.array(list(chain.from_iterable(imSr)))

  ###------------------------- calculation R-Squared betwwen LR and HR ---------###
  r_squared_lr_hr_tot = []
  for i in range(115):
      r_squared_lr_hr = r2_score(np.array(list(chain.from_iterable(validHr[i,:,:,0]))), \
                          np.array(list(chain.from_iterable(validLr_resample[i,:,:,0]))))
      r_squared_lr_hr_tot.append(r_squared_lr_hr)
      
  r_squared_lr_hr_tot = np.array(r_squared_lr_hr_tot)  

  # --------------------- calculation R-squared between HR and SR --------------------------
  r_squared_hr_sr_tot = []
  for i in range(115):
      r_squared_hr_sr = r2_score(np.array(list(chain.from_iterable(validHr[i,:,:,0]))), \
                          np.array(list(chain.from_iterable(images_sr[i][0,:,:,0]))))
      r_squared_hr_sr_tot.append(r_squared_hr_sr)
      
  r_squared_hr_sr_tot = np.array(r_squared_hr_sr_tot) 

  #------------------------------ plot R-Squared LR and HR ---------------------
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cm_to_inch(30),cm_to_inch(10)), dpi=500)

  ax1.plot(imHr_Flat,imLr_Flat, linestyle='none',markerfacecolor='skyblue', marker="o", markeredgecolor="black", markersize=5)
  ax1.set_title('R-squared between LR and HR')
  ax1.set(xlabel='HR', ylabel='LR')
  ax1.set_xlim([-0.5, 0.8])
  ax1.set_ylim([-0.5, 0.8])
  ax1.text(0,  0.6, 'R2=%s'%("{:.3f}".format(r_squared_lr_hr_tot[img_num])) ) 

  ###---------------------------- Plot R-Squared between HR and SR -----------###
  ax2.plot(imSr_Flat,imHr_Flat, linestyle='none',markerfacecolor='skyblue', marker="o", markeredgecolor="black", markersize=5)
  ax2.set_title('R-squared between HR and SR')
  ax2.set(xlabel='HR', ylabel='Generated')
  ax2.set_xlim([-0.5, 0.8])
  ax2.set_ylim([-0.5, 0.8])
  r_squared_value = r2_score(imHr_Flat,imSr_Flat)     # R squared between HR and SR
  ax2.text(0,  0.6, 'R2=%s'%("{:.3f}".format(r_squared_hr_sr_tot[img_num])))

  ###  -----------------calculation of statistics ---------------------------###
  max_rsqrd_lr_hr= np.max(r_squared_lr_hr_tot)
  mean_rsqrd_lr_hr = np.mean(r_squared_lr_hr_tot)  
  
  max_rsqrd_hr_sr= np.max(r_squared_hr_sr_tot)
  mean_rsqrd_hr_sr = np.mean(r_squared_hr_sr_tot) 

  fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(cm_to_inch(30),cm_to_inch(10)), dpi=500)
  im = ax1.imshow(validLr[img_num,:,:,0], cmap='gist_ncar', vmin=0, vmax =+1)
  ax1.set_title('Low resolution')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
  ax1.xaxis.set_visible(False)
  ax1.yaxis.set_visible(False)

  im = ax2.imshow(validHr[img_num,:,:,0], cmap='gist_ncar', vmin=-1, vmax=+1)
  ax2.set_title('High resolution')
  divider = make_axes_locatable(ax2)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
  ax2.xaxis.set_visible(False)
  ax2.yaxis.set_visible(False)

  im = ax3.imshow(images_sr[img_num][0,:,:,0], cmap='gist_ncar', vmin=-1, vmax=+1)
  divider = make_axes_locatable(ax3)
  ax3.set_title('Generated (SR)')
  cax = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(im, cax=cax, orientation='vertical')
  ax3.xaxis.set_visible(False)
  ax3.yaxis.set_visible(False)
  plt.show()

  print(f'R-squard max between low resolution and high resolution is:{max_rsqrd_lr_hr}')
  print(f'R-squared mean between low resolution and high resolution is:{mean_rsqrd_lr_hr}') 

  print(f'R-squard max between high resolution and generated (SR) is:{max_rsqrd_hr_sr}') 
  print(f'R-squared mean between high resolution and generated (SR) is:{mean_rsqrd_hr_sr}')
  return r_squared_lr_hr_tot,r_squared_hr_sr_tot, images_sr