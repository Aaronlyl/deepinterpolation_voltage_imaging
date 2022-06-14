import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

os.chdir('../figures')

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl


#Figure 6
img_path = '../data/data_h5/50.h5'

data, Fd = open_h5(img_path) 

x, D, _,_,_ = get_fl(data, Fd, p=4, start_skip=5)


plt.figure()
plt.plot(x,D,label='raw')
D[D<0]=0
ma = np.convolve(D, np.ones(5)/5, mode='valid')

plt.plot(x[4:],ma,label='envelope',color='black')

plt.xlabel('time (s)')
plt.ylabel('$\Delta$F/F (%)')
plt.xlim([0, 2.5])
plt.legend()

plt.show()
plt.savefig('6.svg')