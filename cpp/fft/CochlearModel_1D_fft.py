import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.fft import dct, idct
from time import time

class params_cochlea:
    def __init__(self, Nx, gamma):
        self.N = Nx
        self.Lb = 3.5
        self.W = 0.1
        self.H = 0.1
        self.b = 0.4
        self.rho = 1.0
        self.dx = self.Lb/self.N
        self.x = np.arange(0,self.Lb,self.dx)

        ch_damp = 2#2.8*np.exp(-0.2*self.x)
        
        self.k1 = 2.2e8*np.exp(-3*self.x)
        self.m1 = 3e-3
        self.c1 = 6 + 670*np.exp(-1.5*self.x) * ch_damp
        self.k2 = 1.4e6*np.exp(-3.3*self.x)
        self.c2 = 4.4*np.exp(-1.65*self.x) * ch_damp
        self.m2 = 0.5e-3
        self.k3 = 2.0e6*np.exp(-3*self.x)
        self.c3 = 0.8*np.exp(-0.6*self.x) * ch_damp
        self.k4 = 1.15e8*np.exp(-3*self.x)
        self.c4 = 440.0*np.exp(-1.5*self.x) * ch_damp

        self.c1c3 = self.c1 + self.c3
        self.k1k3 = self.k1 + self.k3
        self.c2c3 = self.c2 + self.c3
        self.k2k3 = self.k2 + self.k3

        self.gamma = gamma

        self.g = 1
        self.b = 0.4
        self.gamma = gamma

        self.beta = 50e-7

def Gohc(uc, beta):
    return beta*np.tanh(uc/beta)

def dGohc(uc, vc, beta):
    return vc/np.cosh(uc)**2

def get_g(pc, vb, ub, vt, ut):

    gb = pc.c1c3*vb + pc.k1k3*ub - pc.c3*vt - pc.k3*ut
    gt = - pc.c3*vb - pc.k3*ub + pc.c2c3*vt + pc.k2k3*ut

    uc_lin = ub - ut
    vc_lin = vb - vt

    uc = Gohc(uc_lin, pc.beta)
    vc = dGohc(uc_lin, vc_lin, pc.beta)

    gb -= pc.gamma * ( pc.c4*vc + pc.k4*uc )

    return gb, gt

#if __name__ == '__main__':
def solve_time_domain(Nx, f):
    dt = 10e-6
    Ntime = int(round(f.size/2))
    T = Ntime * dt

    t2 = np.arange(0,T,dt/2)
    t = np.arange(0,T,dt)

    g = 0.8
    gamma = np.ones(Nx)*g
    pc = params_cochlea(Nx, gamma)

    Tpre_start = time()

    alpha2 = 4*pc.rho*pc.b/pc.H/pc.m1

    kx = np.arange(1,Nx+1)
    ax = np.pi*(2*kx-1)/4/Nx
    mwx = -4*np.sin(ax)**2/pc.dx**2

    vb = np.zeros((Ntime,Nx))
    ub = np.zeros((Ntime,Nx))
    vt = np.zeros((Ntime,Nx))
    ut = np.zeros((Ntime,Nx))

    p = np.zeros((Ntime,Nx))

    phat = np.zeros(Nx)
    Tpre = time() - Tpre_start

    Tmain_start = time()

    for ii in tqdm.tqdm(range(Ntime-1)):
        ######### RK4 ##################

        # (ii)
        gb, gt = get_g(pc, vb[ii], ub[ii], vt[ii], ut[ii])

        k = -alpha2*gb
        k[0] -= f[ii*2] * 2/pc.dx
        
        #(iii)
        khat = dct(k, type=3)
        phat = khat/(mwx-alpha2)
        p[ii] = idct(phat, type=3)

        #(iv)-(v)
        dvb1 = (p[ii]-gb)/pc.m1 
        ub1 = ub[ii] + 0.5*dt*vb[ii]
        vb1 = vb[ii] + 0.5*dt*dvb1

        dvt1 = -gt/pc.m2
        ut1 = ut[ii] + 0.5*dt*vt[ii]
        vt1 = vt[ii] + 0.5*dt*dvt1    
        
        # (ii)
        gb, gt = get_g(pc, vb1, ub1, vt1, ut1) 

        k = -alpha2*gb
        k[0] -= f[ii*2+1] * 2/pc.dx
        print(k[0])
        #(iii)

        khat = dct(k, type=3)
        phat = khat/(mwx-alpha2)
        p1 = idct(phat, type=3)

        #(iv)-(v)
        dvb2 = (p1-gb)/pc.m1
        ub2 = ub[ii] + 0.5*dt*vb1
        vb2 = vb[ii] + 0.5*dt*dvb2

        dvt2 = -gt/pc.m2
        ut2 = ut[ii] + 0.5*dt*vt1
        vt2 = vt[ii] + 0.5*dt*dvt2   

        # (ii)
        gb, gt = get_g(pc, vb2, ub2, vt2, ut2)

        k = -alpha2*gb
        k[0] -= f[ii*2+1] * 2/pc.dx

        #(iii)

        khat = dct(k, type=3)
        phat = khat/(mwx-alpha2)
        p2 = idct(phat, type=3)

        #(iv)-(v)
        dvb3 = (p2-gb)/pc.m1
        ub3 = ub[ii] + dt*vb2 
        vb3 = vb[ii] + dt*dvb3

        dvt3 = -gt/pc.m2
        ut3 = ut[ii] + dt*vt2
        vt3 = vt[ii] + dt*dvt3  

        # (ii)
        gb, gt = get_g(pc, vb3, ub3, vt3, ut3)
        
        k = -alpha2*gb
        k[0] -= f[ii*2+2] * 2/pc.dx

        #(iii)

        khat = dct(k, type=3)
        phat = khat/(mwx-alpha2)
        p3 = idct(phat, type=3)

        #(iv)-(v)
        dvb4 = (p3-gb)/pc.m1

        dvt4 = -gt/pc.m2  

        ub[ii+1] = ub[ii] + dt/6*(vb[ii] + 2*vb1 + 2*vb2 + vb3)
        vb[ii+1] = vb[ii] + dt/6*(dvb1 + 2*dvb2 + 2*dvb3 + dvb4) 
        ut[ii+1] = ut[ii] + dt/6*(vt[ii] + 2*vt1 + 2*vt2 + vt3)
        vt[ii+1] = vt[ii] + dt/6*(dvt1 + 2*dvt2 + 2*dvt3 + dvt4)

    Tmain = time() - Tmain_start

    return vb, ub, p, Tpre, Tmain

if __name__ == "__main__":
    dt = 10e-6
    T = 20e-6

    Nx = 500

    fp = 1000
    w = 2*np.pi*fp

    tscale = np.arange(0,T,dt)
    tscale2 = np.arange(0,T,dt/2)

    for Lp in [0]:# range(0,120,20):
        Ap = 20e-6 * 10**(Lp/20)

        sinewave = np.sin(w*tscale2) #*Ap*w
        
        vb, ub, p, Tpre, Tmain = solve_time_domain( Nx, sinewave)

   #     pc0 = params_cochlea(Nx, np.zeros(Nx))
   #     plt.plot(pc0.x, 20*np.log10(np.max(np.abs(vb[int(round(tscale.size*9/10)):]), axis=0)))
    plt.show()