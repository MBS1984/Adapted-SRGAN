# from psnr import psnr
import numpy as np
import math
import math

def psnr(img1, img2): 
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0   
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def psnr_func(validHr, validLr, images_sr,images_bicubic):

  psnr_hr_sr_tot =[]
  for i in range(len(validHr)):
                psnr_hr_sr = psnr(validHr[i], images_sr[i])
                psnr_hr_sr_tot.append(psnr_hr_sr)
  psnr_hr_sr_tot = np.array(psnr_hr_sr_tot)
                
  max_psnr_hr_sr = np.max(psnr_hr_sr_tot) 
  mean_psnr_hr_sr = np.mean(psnr_hr_sr_tot)

  print(f'psnr max between hr-sr is:{max_psnr_hr_sr}')
  print(f'psnr mean between hr-sr is:{mean_psnr_hr_sr}')

  #----------- PSNR between HR and Bicubic images  -------- ###
  psnr_HR_bicubic_tot =[]
  for i in range(len(images_bicubic)):
                psnr_HR_bicubic = psnr(validHr[i], images_bicubic[i])
                psnr_HR_bicubic_tot.append(psnr_HR_bicubic)
  psnr_HR_bicubic_tot = np.array(psnr_HR_bicubic_tot)  
              
  max_psnr_hr_bicubic = np.max(psnr_HR_bicubic_tot) 
  mean_psnr_bicubic = np.mean(psnr_HR_bicubic_tot)

  print(f'psnr max between hr-Bicubic is:{max_psnr_hr_bicubic}')
  print(f'psnr mean between hr-Bicubic is:{mean_psnr_bicubic}')
  return psnr_hr_sr_tot, psnr_HR_bicubic_tot