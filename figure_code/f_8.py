import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

os.chdir('../figures')

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl


path1 = '../data/data_h5/50.h5'

path2 = '../data/data_denoised/50_denoised.h5'

data1, Fd = open_h5(path1)

data2 = open_h5(path2, denoised=True) 

x1, d1, _,_,_ = get_fl(data1, Fd,p=4)
x2, d2, _,_,_ = get_fl(data2, Fd,p=4)

fig, ax = plt.subplots(2,1,sharex=True,sharey=True)

ax[0].plot(x1,d1*100,label='raw')
ax[1].plot(x2,d2*100,label='denoised')
ax[1].set_xlabel('time (s)')
ax[0].set_ylabel('$\Delta F / F$ (%)')
fig.legend(loc='upper right',ncol=2)

plt.savefig('8.svg')


