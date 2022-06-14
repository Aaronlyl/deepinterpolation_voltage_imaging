import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

os.chdir('../figures')
sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl

in_dir = '../data/data_h5/'

fr=100
x1 = np.arange(5/fr,250/fr, 1/fr)

fig, axs = plt.subplots(3,2,sharex=True,sharey=True)

with h5py.File(in_dir+'7.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])[5:]


_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
axs[0,0].plot(x1, detrended_in*100, label=(str(7)))

with h5py.File(in_dir+'27.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])[5:]

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
axs[1,0].plot(x1, detrended_in*100, label=(str(27)))

with h5py.File(in_dir+'27.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])[5:]

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
axs[2,0].plot(x1, detrended_in*100, label=(str(28)))
              
with h5py.File(in_dir+'50.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])[5:]


_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
axs[0,1].plot(x1, detrended_in*100, label=(str(50)))

with h5py.File(in_dir+'64.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])[5:]

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
axs[1,1].plot(x1, detrended_in*100, label=(str(64)))

with h5py.File(in_dir+'114.h5','r') as hf:
    Fd = np.array(hf["Fd"])
    img_in = np.array(hf["data"])[5:]

_,detrended_in, tSNR, _, act_map_in =get_fl(img_in, Fd=Fd,p=3)
axs[2,1].plot(x1, detrended_in*100, label=(str(114)))
              
fig.legend(title='file number', ncol=2,loc='upper right')

for i in range(3):
    for j in range(2):
        axs[i,j].spines['top'].set_visible(False)
        axs[i,j].spines['right'].set_visible(False)
        
axs[1,0].set_ylabel('$\Delta F/F$ (%)')
axs[2,0].set_xlabel('time (s)')
axs[2,1].set_xlabel('time (s)')

plt.savefig('13.svg')