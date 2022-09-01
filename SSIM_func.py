import tensorflow as tf
import numpy as np
import cv2
### --- calculation of the stractural similarilty Index between HR and SR ---###
def SSIM_func(validLr, validHr, images_sr):
  
  images_hr_tnsr = tf.convert_to_tensor(validHr)

  images_sr_1 = np.array(images_sr)
  images_sr = images_sr_1[:,0,:,:]
  images_sr_tnsr = tf.convert_to_tensor(images_sr)

  ssim_hr_sr = tf.image.ssim(images_hr_tnsr,images_sr_tnsr, max_val=2 )
  ssim_hr_sr  = np.array(ssim_hr_sr)
  

  images_bicubic  = []
  for i in range(validLr.shape[0]):
      image_bicubic = cv2.resize(validLr[i,:,:,0], dsize=(validHr.shape[1], validHr.shape[2]), interpolation=cv2.INTER_CUBIC)
      images_bicubic.append(image_bicubic)
    
  images_bicubic = np.array(images_bicubic)
  images_bicubic = images_bicubic[:,:,:,np.newaxis]
  images_bicubic = (images_bicubic * 2.) -1 
    
  # SSIM between HR and bicubic images
  images_bicubic_tnsr = tf.convert_to_tensor(images_bicubic)
  ssim_hr_bicubic = tf.image.ssim(images_hr_tnsr, images_bicubic_tnsr, max_val=1) 
  ssim_hr_bicubic  = np.array(ssim_hr_bicubic)
    
  max_ssim_HR_bicubic = np.max(ssim_hr_bicubic)
  mean_ssim_HR_bicubic = np.mean(ssim_hr_bicubic)
  
  max_ssim_hr_sr = np.max(ssim_hr_sr)
  mean_ssim_hr_sr = np.mean(ssim_hr_sr)
  
  print(f'Maximum ssim between high resolution and bicubic is : {max_ssim_HR_bicubic}')
  print(f'Mean ssim between high resolution and bicubic is : {mean_ssim_HR_bicubic}')
  
  print(f'Maximum ssim between high resolution and generated (SR) is : {max_ssim_hr_sr}')
  print(f'Mean ssim between high resolution and generated (SR) is : {mean_ssim_hr_sr}')
  return images_bicubic, ssim_hr_sr, ssim_hr_bicubic