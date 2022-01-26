import numpy as np
import matplotlib.pyplot as plt

filename = 'mat_vb.dat'
x = np.fromfile(filename, dtype=np.float64)
x = x.reshape((10000,500)) # Time, Number of segments

plt.plot(x[-1,:])
plt.show()
