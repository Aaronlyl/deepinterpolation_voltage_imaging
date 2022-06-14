import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

os.chdir('../figures')

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl

def plot_compare(data_num, raw_dir, denoised_dir, ax):
    
    with h5py.File(raw_dir+ str(data_num)+".h5",'r') as hf:
        Fd = np.array(hf["Fd"])
        img_in = np.array(hf["data"])
        img_in = img_in[5:]

    with h5py.File(denoised_dir+ str(data_num)+'_denoised.h5','r') as hf:
        img_out = np.array(hf['data'])
    
    _,detrended_in, _, _, _=get_fl(img_in, Fd=Fd,p=3)
    _,detrended_out, _,_, _ = get_fl(img_out, Fd=Fd,p=3, denoised=True)
    
    fr = 100
    # plt.subplot(2,1,1)
    x1 = np.arange(5/fr,250/fr, 1/fr)
    x2 = np.arange(60/fr,60/fr+190/fr, 1/fr)
    ax.plot(x1[0:len(detrended_in)], detrended_in*100,label='raw')
    ax.plot(x2[0:len(detrended_out)], detrended_out*100,label='denoised')
    ax.set_xlim([0, 2.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True, sharey=True)
raw_dir = '../data/data_h5/'
denoised_dir = '../data/data_denoised/'
plot_compare(50, raw_dir, denoised_dir, ax1)
    
plot_compare(83, raw_dir, denoised_dir, ax2)

plot_compare(118, raw_dir, denoised_dir, ax3)
fig.legend(['raw','denoised'], loc = "lower center", ncol=5 )
ax3.set_xlabel('time (s)')
ax2.set_ylabel('$\Delta F/F$ (%)')
plt.tight_layout()  
plt.savefig('9.svg')
