import numpy as np
import scipy.io.wavfile as sio
from scipy.signal import resample
import matplotlib.pyplot as plt

def load(filename, Lpeak):
    
    fs_model = 200e3
    fs, data = sio.read(filename)


    dt = 1/fs
    T = data.size/fs

    numT = data.size

    data = data.astype(np.float64)
    data = data/np.max(np.abs(data))


    data = resample(data, int(round(numT*fs_model/fs)))
    
    multi = 20e-5*10**(Lpeak/20.0)

    signal = np.zeros(data.shape)
    signal[1:-1] = multi*(data[2:]-data[:-2])/2/dt

    return signal