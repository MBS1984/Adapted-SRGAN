# -*- coding: utf-8 -*-
"""
Function for visualization of different super-resolution method

"""
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


MISSING = object()

def create_cmap(col1, col2 = (0,0,0)):
    newcolors = np.zeros((256,4))
    new_col1 = np.array(list(col1) + [1])
    new_col2 = np.array(list(col2) + [1])
    newcolors[int(256/2):, :] = new_col1
    newcolors[:int(256/2), :] = new_col2
    return ListedColormap(newcolors)

def import_model(name,
                 input_size,
                 filter_size = (3,3),
                 trained = MISSING): 
    """
     function to import the desired model
     in: name - name of the model to load
           input_size - the size of the input 
           filter_size - filter size (DEFAULT: 3 by 3)
           trained: the pretrained weights (DEFAULT = MISSING)
    """
    md =importlib.import_module(name)
    model = md.model(input_size = input_size,
                     filter_size=filter_size)
    
    if trained is not MISSING:
        model.load_weights(trained)    
    return model

def cm_to_inch(value):
    return value/2.54

def compare_pred(image_low, image_high, model, name = '', save = False, cmap = 'gist_ncar',tile_number=''):
    model.eval()
    
    image0 = image_low.reshape(1, image_low.shape[0], image_low.shape[1], 1)
    image1 = image_high.reshape(1, image_high.shape[0], image_high.shape[1], 1)
    image2 = model(image0).numpy()
    image3 = cv2.resize(image0[0,:,:,0], dsize=(image_high.shape[0], image_high.shape[1]), interpolation=cv2.INTER_CUBIC)
 
    image0 = image0[0, :, :, 0]
    image1 = image1[0, :, :, 0]
    image2 = image2[0, :, :, 0]
    image3 = image3
      
    fig, axes = plt.subplots(1 , 4, figsize=(cm_to_inch(40),cm_to_inch(10)), dpi=100)
    
    # low_resolution
    b = axes.flat[0] 
    im = b.imshow(image0, cmap = 'gist_ncar', vmin = 0,vmax=1)
    divider = make_axes_locatable(b)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')      
    b.set_title('Low Resolution' '#%s'%(format(tile_number)),fontname='Comic Sans MS', fontsize = 12)
    b.spines['left'].set_visible(False)
    b.spines['right'].set_visible(False)
    b.spines['bottom'].set_visible(False)
    b.spines['top'].set_visible(False)
    b.xaxis.set_visible(False)
    b.yaxis.set_visible(False)
    
    
    # high resolution
    a = axes.flat[1] 
    im = a.imshow(image1,cmap = 'gist_ncar', vmin = -1,vmax=1)
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')  
    a.set_title('High Resolution',fontname='Comic Sans MS', fontsize = 12)
    a.spines['left'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.xaxis.set_visible(False)
    a.yaxis.set_visible(False)
    
    # Super resolution (Generated)
    d = axes.flat[2]
    im = d.imshow(image2,cmap = 'gist_ncar',  vmin = -1, vmax=1)
    divider = make_axes_locatable(d)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')     
    d.set_title('Generated', fontname='Comic Sans MS', fontsize = 12)
    d.spines['left'].set_visible(False)
    d.spines['right'].set_visible(False)
    d.spines['bottom'].set_visible(False)
    d.spines['top'].set_visible(False)
    d.xaxis.set_visible(False)
    d.yaxis.set_visible(False)
    
    # Bicubic interpolation
    c = axes.flat[3] 
    im = c.imshow(image3, cmap = 'gist_ncar', vmin = 0, vmax=1)
    divider = make_axes_locatable(c)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    c.set_title('Bicubic interpolation',fontname='Comic Sans MS', fontsize = 12)
    c.spines['left'].set_visible(False)
    c.spines['right'].set_visible(False)
    c.spines['bottom'].set_visible(False)
    c.spines['top'].set_visible(False)
    c.xaxis.set_visible(False)
    c.yaxis.set_visible(False)
    
    if save:
        plt.tight_layout()
        plt.savefig("/content/drive/MyDrive/Colab Notebooks/ASRGAN/samples/tiles_%s.pdf"%(name), transparent=False) 
        
    plt.show()
    return image0, image3
    
def load_G(g=MISSING):
    G = get_G((1, 48, 48, 1))
    if g is not MISSING:
        G.load_weights(g)
    return G

