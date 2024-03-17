import numpy as np
from scipy.fft import dct, idct
import time

N = 512
dtype = np.float64
def myDOT():

    A = np.zeros((N,N), dtype=dtype)
    x = np.zeros(N, dtype=dtype)

    for nn in range(800000):
        b = np.dot(A,x)

def myDCT():

    x = np.zeros(N, dtype=dtype)
    for nn in range(800000):
        y = dct(x)
        z = idct(y)

if __name__ == "__main__":

    N = 512

    print("%s %s"%(N, dtype))
    start = time.time()
    myDOT()
    elapsed_time = time.time() - start

    print("DOT: "+str(elapsed_time))
    
    start = time.time()
    myDCT()
    elapsed_time = time.time() - start
    
    print("DCT: "+str(elapsed_time))