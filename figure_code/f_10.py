import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl

path1 = '../data/data_h5/50.h5'
path2 = '../data/data_denoised/50_denoised.h5'



img_in, Fd = open_h5(path1)
img_out = open_h5(path2,denoised=True)

_,_,_,_,act_map_in = get_fl(img_in, Fd=Fd, p=3)
_,_,_,_,act_map_out = get_fl(img_out, Fd=Fd, p=3, denoised=True)


dim = 532.48/2
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img_in[0],extent=[0, dim, 0, dim])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.text(10, 20, 'snapshot: raw', bbox={'facecolor': 'white', 'pad': 2})
plt.subplot(2,2,2)
plt.imshow(img_out[0],extent=[0, dim, 0, dim])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.text(10, 20, 'snapshot: denoised', bbox={'facecolor': 'white', 'pad': 2})


plt.subplot(2,2,3)
plt.imshow(act_map_in,extent=[0, dim, 0, dim])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.text(10, 20, 'activation map: raw', bbox={'facecolor': 'white', 'pad': 2})
plt.subplot(2,2,4)
plt.imshow(act_map_out,extent=[0, dim, 0, dim])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.text(10, 20, 'activation map: denoised', bbox={'facecolor': 'white', 'pad': 2})

plt.show()
plt.savefig('10.svg')