import torch
import numpy as np
import math
def psnr(target, ref):
    target = torch.tensor(target)
    ref = torch.tensor(ref)
    diff = torch.square(target - ref).float()
    diff = torch.sqrt(torch.mean(diff))
    PSNR = 20*torch.log(255./(diff+0.001))
    return PSNR

 
def numpy_psnr(target, ref, scale):
    target_data = np.array(target)
    target_data = target_data[scale:-scale,scale:-scale]
 
    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale,scale:-scale]
 
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)