import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl

os.chdir('../figures')

in_dir = '../data/data_h5/'
out_dir = '../data/data_denoised/'

nums = [7, 27, 28, 29, 50, 60, 64, 83, 100, 114, 118]

tSNR_in = np.zeros(len(nums))
tSNR_out = np.zeros(len(nums))
noise_in = np.zeros(len(nums))
noise_out = np.zeros(len(nums))

for i in range(len(nums)):
    img_in, Fd = open_h5(in_dir+str(nums[i])+'.h5')

    img_out = open_h5(out_dir+str(nums[i])+'_denoised.h5',denoised=True)
   
    _, _,tSNR_in[i], noise_in[i], _= get_fl(img_in, Fd=Fd, p=4)
    _, _,tSNR_out[i], noise_out[i], _= get_fl(img_out, Fd=Fd, p=4, denoised=True)
    

# print('mean denoised tSNR: ' +np.mean(tSNR_out))
# print('mean raw tSNR: ' + np.mean(tSNR_in))
# print('average tSNR percentage change: ' + (np.mean(tSNR_out) - np.mean(tSNR_in))/np.mean(tSNR_in))
# print('average noise percentage change: ' + ((np.mean(noise_out)-np.mean(noise_in))/np.mean(noise_in)))

x = ['raw', 'deepinterpolation']
fig, (ax1, ax2) = plt.subplots(1,2)
# plt.subplot(1,2,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.plot(x, [tSNR_in, tSNR_out],'-o', label=nums)
ax1.set_ylabel('tSNR')
ax1.set_title('tSNR')
ax1.legend()

plt.subplot(1,2,2)
ax2.plot(x, [noise_in*100, noise_out*100],'-o',label=nums)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel('noise (%)')
ax2.set_title('Noise')
ax2.legend()
plt.tight_layout()
plt.savefig('11.svg')

