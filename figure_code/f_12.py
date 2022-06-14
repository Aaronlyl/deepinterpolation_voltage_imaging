import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

os.chdir('../figures')

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl

fr = 100
nums = [7, 27, 28, 29, 50, 60, 64, 83, 100, 114, 118]
x1 = np.arange(5/fr,250/fr, 1/fr)
x2 = np.arange(60/fr,60/fr+190/fr, 1/fr)

raw_dir = '../data/data_h5/'
denoised_dir = '../data/data_denoised/'


fig, axs = plt.subplots(3,2, sharex=True)

with h5py.File(raw_dir+str(nums[0])+'.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])
img_in = img_in[5:]
print(img_in.shape)

with h5py.File(denoised_dir+str(nums[0])+'_denoised.h5','r') as hf:
    img_out = np.array(hf['data'])

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
_,detrended_out, tSNR, _, act_map_out = get_fl(img_out, Fd=Fd,p=3, denoised=True)



axs[0,0].plot(x1[0:len(detrended_in)], detrended_in*100,label='raw',alpha=0.7)
axs[0,1].plot(x2[0:len(detrended_out)], detrended_out*100,label='denoised', color='black')
axs[0,0].spines['right'].set_visible(False)
axs[0,0].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)
axs[0,1].spines['top'].set_visible(False)



with h5py.File(raw_dir+str(nums[1])+'.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])
img_in = img_in[5:]
print(img_in.shape)

with h5py.File(denoised_dir+str(nums[1])+'_denoised.h5','r') as hf:
    img_out = np.array(hf['data'])

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
_,detrended_out, tSNR, _, act_map_out = get_fl(img_out, Fd=Fd,p=3, denoised=True)

axs[1,0].plot(x1[0:len(detrended_in)], detrended_in*100,label='raw',alpha=0.7)
axs[1,1].plot(x2[0:len(detrended_out)], detrended_out*100,label='denoised', color='black')
axs[1,0].spines['right'].set_visible(False)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].set_ylabel('$\Delta F/F$ (%)')
axs[1,1].spines['right'].set_visible(False)
axs[1,1].spines['top'].set_visible(False)



with h5py.File(raw_dir+str(nums[2])+'.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])
img_in = img_in[5:]
print(img_in.shape)

with h5py.File(denoised_dir+str(nums[2])+'_denoised.h5','r') as hf:
    img_out = np.array(hf['data'])

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
_,detrended_out, tSNR, _, act_map_out = get_fl(img_out, Fd=Fd,p=3, denoised=True)

axs[2,0].plot(x1[0:len(detrended_in)], detrended_in*100,label='raw',alpha=0.7)
axs[2,1].plot(x2[0:len(detrended_out)], detrended_out*100,label='denoised', color='black')
axs[2,0].set_xlabel('time(s)')
axs[2,0].spines['right'].set_visible(False)
axs[2,0].spines['top'].set_visible(False)
axs[2,0].set_xlim([0, 2.5])
axs[2,1].spines['right'].set_visible(False)
axs[2,1].spines['top'].set_visible(False)
axs[2,1].set_xlabel('time (s)')


fig.legend(['raw','denoised'],loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig('12.svg')