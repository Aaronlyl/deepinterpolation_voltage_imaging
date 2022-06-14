import numpy as np


# get TCC by scanning for the maximum, then take the local_mean (+- 2 frames) to find tau
def get_TCC(dF_over_F_d): 
    plt.plot(dF_over_F_d)
   
    tcc = 0
    peak_idx = np.argmax(dF_over_F_d)
    peak = np.max(dF_over_F_d)
    for t in np.arange(peak_idx, len(dF_over_F_d)-30):
        if dF_over_F_d[t] < (1/3)*peak:
            if np.mean(dF_over_F_d[t-2:t+2])<(1/3)*peak:
                tcc = t - peak_idx
                return tcc
    return tcc