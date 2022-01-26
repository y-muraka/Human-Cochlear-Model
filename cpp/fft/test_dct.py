import numpy as np
from scipy.fft import dct

FFT_LEN = 512

x = np.random.random(FFT_LEN)
x_dct = dct(x,type=1)
x_inv = dct(x_dct,type=1)/FFT_LEN/2

print("Original:")
print((x[:5]))
print("DCT:")
print((x_dct[:5]))
print("Inverse:")
print((x_inv[:5]))