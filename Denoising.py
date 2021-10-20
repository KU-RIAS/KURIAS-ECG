get_ipython().run_line_magic('reset', '')
import sys
sys.path.append('"User_directory"\rampy-master\rampy')
#### Ref #######
#### Reference: Baseline correction using asymmetrically reweighted penalized least squares smoothing. Baek et al. 2015, Analyst 140: 250-257;
#### Code reference: https://github.com/charlesll/rampy
import numpy as np
import pandas as pd
import rampy as rp
import pickle
from scipy.signal import butter, lfilter

# Open ECG signal data
with open('"User_directory"/"file_name".pkl', 'rb') as f:
    data_all = pickle.load(f)

for num_wave in range(0,12): # Range(0,12) means the column of the signal.
    ECG_raw = []
    ECG_raw = data_all.iloc[:,num_wave]  # Each ECG signal
    ECG_split = ECG_raw.str.split(',',expand=True)  # Split of comma type
    ECG_siganl = ECG_split.astype(float)
    ECG_number = len(ECG_siganl)

# Define butterworth_bandpass filter
    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
# Sample rate and desired cutoff frequencies
    fs = 500.0
    lowcut = 0.05
    highcut = 150.0
    T = 1/fs
    
# Apply Bandpass
    Filtered_data = []
    
    for i in range(ECG_number):
          Sample_signal = ECG_siganl.iloc[i,:]
          Sample_samples = int(T * fs)
          Sample_t = np.linspace(0, T, Sample_samples, endpoint=False)
          y = butter_bandpass_filter(Sample_signal, lowcut, highcut, fs, order=4)
          Filtered_data.append(np.array(np.reshape(y,(5000,))))
    
    DF_filtered = pd.DataFrame(Filtered_data)
    
# Apply basline filter(arPLS)    
    x = np.arange(0,5000,1.0)
    np_filtered = np.array(Filtered_data)
    Obs_corr = np.ones(np_filtered.shape)
    Bas_total= np.ones(np_filtered.shape)
    ROI = np.array([[0.,1000.],[4000.,5000.]])
    
    for i in range(np_filtered.shape[0]):
        sig_corr, bas_, = rp.baseline(x, np_filtered[i,:].T,ROI, lam = 10**6, method="arPLS")
        Obs_corr[i,:] = sig_corr.reshape(1,-1)
        Bas_total[i,:] = bas_.reshape(1,-1)
        sig_corr = []
        bas_ = [] 
        
# Save
# =============================================================================
#     df_obs = pd.DataFrame(Obs_corr)
#     df_obs.to_csv('"User_directory"\"file_name"_{}.csv'.format(number),index = False)
#     DF_filtered = []
#     Obs_corr =[]
# =============================================================================
