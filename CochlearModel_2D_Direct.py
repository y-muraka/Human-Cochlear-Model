import numpy as np
import matplotlib.pyplot as plt
import tqdm
import wavfile

class CochlearModel:
    def __init__(self, Nx, Ny, gamma):
        self.Nx = Nx
        self.Ny = Ny
        self.Lb = 3.5
        self.L = 0.1
        self.W = 0.1
        self.H = 0.1
        self.b = 0.4
        self.rho = 1.0
        self.dx = self.Lb/self.Nx
        self.dy = self.H/(self.Ny-1)
        self.x = np.arange(0,self.Lb,self.dx)
        self.y = np.linspace(0,self.H,self.Ny)

        ch_damp = 2
        
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

        self.g = 1
        self.b = 0.4
        self.gamma = gamma

        self.dt = 10e-6

        self.beta = 50e-7

    def Gohc(self, uc, beta):
        return beta*np.tanh(uc/beta)

    def dGohc(self, uc, vc, beta):
        return vc/np.cosh(uc)**2

    def get_g(self, vb, ub, vt, ut):

        gb = self.c1c3*vb + self.k1k3*ub - self.c3*vt - self.k3*ut
        gt = - self.c3*vb - self.k3*ub + self.c2c3*vt + self.k2k3*ut

        uc_lin = ub - ut
        vc_lin = vb - vt

        uc = self.Gohc(uc_lin, self.beta)
        vc = self.dGohc(uc_lin, vc_lin, self.beta)

        gb -= self.gamma * ( self.c4*vc + self.k4*uc )

        return gb, gt

    def solve_time_domain(self, f):

        Ntime = int(round(f.size/2))
        T = Ntime * self.dt

        t2 = np.arange(0,T,self.dt/2)
        t = np.arange(0,T,self.dt)

        alpha2 = 4*self.rho*self.b/self.dy/self.m1

        vb = np.zeros((Ntime,Nx))
        ub = np.zeros((Ntime,Nx))
        vt = np.zeros((Ntime,Nx))
        ut = np.zeros((Ntime,Nx))

        p = np.zeros((Ntime,Nx))

        Ay = np.zeros((Ny,Ny))
        Ay[0,0] = -2 - alpha2*self.dy**2
        Ay[0,1] = 2 
        for nn in range(1,Ny-1):
            Ay[nn,nn-1] = 1
            Ay[nn,nn] = -2
            Ay[nn,nn+1] = 1
        Ay[Ny-1,Ny-2] = 2 
        Ay[Ny-1,Ny-1] = -2 
        Ay /= self.dy**2

        Iy = np.eye(Ny)
        F = np.zeros((Nx*Ny,Nx*Ny))
        F[:Ny,:Ny] = -2*Iy + Ay*self.dx**2
        F[:Ny,Ny:Ny*2] = 2*Iy
        for mm in range(1,Nx-1):
            F[mm*Ny:(mm+1)*Ny,(mm-1)*Ny:mm*Ny] = Iy 
            F[mm*Ny:(mm+1)*Ny,mm*Ny:(mm+1)*Ny] = -2*Iy  + Ay*self.dx**2 
            F[mm*Ny:(mm+1)*Ny,(mm+1)*Ny:(mm+2)*Ny] = Iy 
        F[(Nx-1)*Ny:,(Nx-2)*Ny:(Nx-1)*Ny] = Iy
        F[(Nx-1)*Ny:,(Nx-1)*Ny:] = -2*Iy + Ay*self.dx**2
        F /= self.dx**2

        iF = np.linalg.inv(F)

        for ii in tqdm.tqdm(range(Ntime-1)):
            ######### RK4 ##################

            # (ii)
            gb, gt = self.get_g(vb[ii], ub[ii], vt[ii], ut[ii])

            k = np.zeros(Nx*Ny)
            k[::Ny] -= alpha2*gb
            k[:Ny] -= f[ii*2] * 2/self.dx
            
            #(iii)
            p[ii] = np.dot(iF, k)[::Ny]

            #(iv)-(v)
            dvb1 = (p[ii]-gb)/self.m1 
            ub1 = ub[ii] + 0.5*self.dt*vb[ii]
            vb1 = vb[ii] + 0.5*self.dt*dvb1

            dvt1 = -gt/self.m2
            ut1 = ut[ii] + 0.5*self.dt*vt[ii]
            vt1 = vt[ii] + 0.5*self.dt*dvt1    
            
            # (ii)
            gb, gt = self.get_g(vb1, ub1, vt1, ut1) 

            k = np.zeros(Nx*Ny)
            k[::Ny] -= alpha2*gb
            k[:Ny] -= f[ii*2+1] * 2/self.dx

            #(iii)
            p1 = np.dot(iF, k)[::Ny]

            #(iv)-(v)
            dvb2 = (p1-gb)/self.m1
            ub2 = ub[ii] + 0.5*self.dt*vb1
            vb2 = vb[ii] + 0.5*self.dt*dvb2

            dvt2 = -gt/self.m2
            ut2 = ut[ii] + 0.5*self.dt*vt1
            vt2 = vt[ii] + 0.5*self.dt*dvt2   

            # (ii)
            gb, gt = self.get_g(vb2, ub2, vt2, ut2)

            k = np.zeros(Nx*Ny)
            k[::Ny] -= alpha2*gb
            k[:Ny] -= f[ii*2+1] * 2/self.dx

            #(iii)
            p2 = np.dot(iF, k)[::Ny]

            #(iv)-(v)
            dvb3 = (p2-gb)/self.m1
            ub3 = ub[ii] + self.dt*vb2 
            vb3 = vb[ii] + self.dt*dvb3

            dvt3 = -gt/self.m2
            ut3 = ut[ii] + self.dt*vt2
            vt3 = vt[ii] + self.dt*dvt3  

            # (ii)
            gb, gt = self.get_g(vb3, ub3, vt3, ut3)
            
            k = np.zeros(Nx*Ny)
            k[::Ny] -= alpha2*gb
            k[:Ny] -= f[ii*2+2] * 2/self.dx

            #(iii)
            p3 = np.dot(iF, k)[::Ny]

            #(iv)-(v)
            dvb4 = (p3-gb)/self.m1

            dvt4 = -gt/self.m2  

            ub[ii+1] = ub[ii] + self.dt/6*(vb[ii] + 2*vb1 + 2*vb2 + vb3)
            vb[ii+1] = vb[ii] + self.dt/6*(dvb1 + 2*dvb2 + 2*dvb3 + dvb4) 
            ut[ii+1] = ut[ii] + self.dt/6*(vt[ii] + 2*vt1 + 2*vt2 + vt3)
            vt[ii+1] = vt[ii] + self.dt/6*(dvt1 + 2*dvt2 + 2*dvt3 + dvt4)

        return vb, ub, p

if __name__ == "__main__":
    Nx = 300
    Ny = 10 
    g = 0.7

    gamma = np.ones(Nx)*g

    cm = CochlearModel(Nx, Ny, gamma)

    Lps = np.arange(0,120,20)

    for fp in [250, 1000, 4000]:
        filename = '%gHz.wav'%(fp)
        plt.figure()
        for Lp in Lps:
            print("%dHz %ddB"%(fp, Lp))
            sinewave = wavfile.load(filename, Lp)

            vb, ub, p = cm.solve_time_domain( sinewave )

            plt.plot(cm.x*10, 20*np.log10(np.max(np.abs(vb[int(round(vb.shape[0]*9/10)):]), axis=0)*10))
        plt.xlabel('Distance from the stapes [mm]')
        plt.ylabel('BM velocity [dB re 1 mm/s]')
        plt.title('%d Hz'%(fp))
    plt.show()