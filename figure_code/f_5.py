import numpy as np
import os
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import h5py
import sys

os.chdir('../figures')

sys.path.append('../helpers')
from plot_helpers import get_F0, rms, get_tSNR, open_h5, get_fl


#Figure 5_1

img_path = '../data/data_h5/50.h5'

data, Fd = open_h5(img_path) 

x1, dF1 ,_,_,_ = get_fl(data, Fd=Fd, p=0)
x2, dF2 ,_,_,_= get_fl(data, Fd=Fd, p=4)

model = np.polyfit(x1, dF1*100, 4)
trend = np.polyval(model, x1)

fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(x1, dF1*100,label='raw')
ax[0].plot(x1, trend,label='trend')
ax[0].grid(True)
ax[0].set_ylabel('$\Delta F/F$')
ax[0].legend()


ax[1].plot(x2, dF2*100)
ax[1].set_xlabel('time (s)')
ax[1].grid(True)

plt.tight_layout()
plt.savefig('../figures/5_1.svg')

#Figure 5_2

in_dir = '../data/data_h5/'

nums = [7, 27, 28, 29, 50, 60, 64, 83, 100, 114, 118]

x = np.arange(2,10)

MS = []

for i in range(len(nums)):

    data, Fd = open_h5(in_dir+str(nums[i])+'.h5')
        
        
    F0 = get_F0(data)
    
    dFF = (data-F0)/ (F0-Fd)
        
    variance = np.var(dFF,axis=0)

    threshold = np.percentile(variance,98)
    binary_map = np.greater(variance, threshold)


    dF_over_F = np.zeros(data.shape[0])
    
                    
    for t in range(len(dF_over_F)):
        tmp_F = []
        for idx, i in np.ndenumerate(data[t]):
            if binary_map[idx]:
                tmp_F.append((i-F0)/(F0-Fd))
        dF_over_F[t] = np.mean(tmp_F)
        
    x = np.arange(0,data.shape[0]/100,1/100)

    ms = np.zeros(len(x))
    for j in range(len(x)):
        
        model = np.polyfit(x, dF_over_F, x[j])
        trend = np.polyval(model, x)
        ms[j] = np.sqrt(np.mean(np.power(dF_over_F-trend,2)))
        
    MS.append(ms)
    

plt.figure()
plt.plot(x,np.mean(MS,axis=0),color='black')
plt.xlabel('order')
plt.ylabel('mean squared error')
plt.title('Avg. mean squared error of polynomial fit')
plt.grid(True)
plt.tight_layout()
plt.savefig('5_2.svg')

