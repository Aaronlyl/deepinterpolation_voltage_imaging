import numpy as np
import os
import matplotlib.pyplot as plt
import h5py

def get_F0(img):
    return np.mean(img[0:9])

def rms(series):
    return np.sqrt(np.mean(np.power(series,2)))

def get_peak(series):
    return np.max(series)

def get_tSNR(data):
    noise = rms(data[0:19])
    peak = get_peak(data[20:len(data)-20])
    return peak/noise, noise


def open_h5(img_path, denoised=False):
    with h5py.File(img_path, 'r') as hf:
        data = np.array(hf['data'])
        if denoised is True:
            return data
        Fd = np.array(hf['Fd'])
        return data, Fd


#Obtain Delta F over F

def get_fl(img, Fd, denoised=False, p=4, ppf=30, fr=100, start_skip=0):
    F0 = get_F0(img)
    T = img.shape[0]
    X = img.shape[1]
    Y = img.shape[2]
    
  
    
    dF = (img-F0) / (F0 - Fd)
    
    var= np.var(dF,axis=0)
    
    threshold = np.percentile(var, 98)
    bin_map = np.greater(var, threshold)
    act_map = np.ma.masked_where(var>threshold, img[0])
    
    dF_over_F = np.zeros(T)
    
    #Get dF_over_F from identified ROI
    for t in range(T):
        tmp_F = []
        for idx, i in np.ndenumerate(img[t]):
            if bin_map[idx]:
                tmp_F.append((i-F0)/(F0-Fd))
        dF_over_F[t] = np.mean(tmp_F)
    
    if denoised is True:
        x = np.arange(ppf/fr, (ppf+T)/fr, 1/fr)
        dF_over_F = -dF_over_F
    else:
        x = np.arange(start_skip/fr, (start_skip+T)/fr, 1/fr)
        
    if p!=0:
        
        model = np.polyfit(x, dF_over_F, p)
        trend = np.polyval(model, x)
    
         # Detrended dF_over_F + invert
        dF_over_F = -(dF_over_F - trend)
    
    
    tSNR, noise = get_tSNR(dF_over_F)
    
    return x, dF_over_F, tSNR, noise, act_map


