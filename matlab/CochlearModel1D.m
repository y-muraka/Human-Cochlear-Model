classdef CochlearModel1D
% Translated from original Python code by rtachi
% The original code was implmented by y-muraka
%
% document copied from original ---------------------------------------
%
%     One-dimensional cochlear model with two-degree-of-freedom
%     (2DOF) micro-structure [1] for human. This program employs 
%     time domain solution proposed in Ref. [2].
% 
%     Ref.
%     [1] Neely S and Kim D, "A model for active elements in cochlear biomechanics,"
%     The Journal of the Acoustical Society of America, 79(5), 1472--1480, 1986.
%     [2] Diependaal, R.J et al, "Numerical methods for solving one-dimensional
%     cochlear models in the time domain, " The Journal of the Acoustical Society of 
%     America, 82 (5), 1655--1666, 1987
%     
%     Attributes
%     ----------
%     N : int
%         Number of segments
%     Lb : float
%         Cochlear length [cm]
%     W : float
%         Witdh of basilar membrane (BM) [cm]
%     H : float
%         Height of BM [cm]
%     b : float
%         ratio of BM to CP displacement
%     rho : float
%         Fluid density [dyn cm^-3]
%     dx : float
%         Spacing between two segments [cm]
%     x : ndarray
%         Longitudial poisition from the stapes [cm]
%     k1 : ndarray
%         Compliance of BM [dyn cm^-3]
%     m1 : ndarray
%         Mass of BM [g cm^-2]
%     c1 : ndarray 
%         Resistance of BM [dyn s cm^-3]
%     k2 : ndarray
%         Compliance of tectrial membrane (TM) [dyn cm^-3]
%     m2 : ndarray
%         Mass of TM [g cm^-2]
%     c2 : ndarray
%         Resistance of TM [dyn s cm^-3]
%     k3 : ndarray
%         Compliance of connection between BM and TM [dyn cm^-3]
%     c3 : ndarray
%         Resistance of connection between BM and TM [dyn s cm^-3]
%     k4 : ndarray
%         Compliance of outer hair cell's (OHC's) activity [dyn cm^-3]
%     c4 : ndarray
%         Resistance of outer hair cell's (OHC's) activity [dyn s cm^-3]
%     gamma : ndarray
%         Gain factor distribution 
%     dt : float
%         Time step for time domain simulation [sec]
%     beta : float
%         Complete saturating point in OHC's active process [cm]
% ------------------------------------------------------------------------
    properties 
        N, Lb, W, H, b, rho, dx, x
        k1, m1, c1, k2, c2, m2, k3, c3, k4, c4, c1c3, k1k3, c2c3, k2k3
        gamma, dt, beta
    end
    
    methods
        % constructor (initializer)
        function self = CochlearModel1D(Nx, gamma) 
        %     Parameters
        %     ----------
        %     Nx : int
        %         Number of segment
        %     gamma : ndarray
        %         Gain factor distribution
            self.N = Nx;
            self.Lb = 3.5;
            self.W = 0.1;
            self.H = 0.1;
            self.b = 0.4;
            self.rho = 1.0;
            self.dx = self.Lb/self.N;
            self.x = (0:Nx-1)*self.dx;

            ch_damp = 2.8*exp(-0.2*self.x);       

            self.k1 = 2.2e8*exp(-3*self.x);
            self.m1 = 3e-3;
            self.c1 = 6 + 670*exp(-1.5*self.x) .* ch_damp;
            self.k2 = 1.4e6*exp(-3.3*self.x);
            self.c2 = 4.4*exp(-1.65*self.x) .* ch_damp;
            self.m2 = 0.5e-3;
            self.k3 = 2.0e6*exp(-3*self.x);
            self.c3 = 0.8*exp(-0.6*self.x) .* ch_damp;
            self.k4 = 1.15e8*exp(-3*self.x);
            self.c4 = 440.0*exp(-1.5*self.x) .* ch_damp;

            self.c1c3 = self.c1 + self.c3;
            self.k1k3 = self.k1 + self.k3;
            self.c2c3 = self.c2 + self.c3;
            self.k2k3 = self.k2 + self.k3;

            self.gamma = gamma;

            self.dt = 10e-6;

            self.beta = 50e-7;
        end
        % utilities
        function output = Gohc(self,uc)
            output =  self.beta*tanh(uc/self.beta);
        end
        function output = dGohc(self, uc, vc)
            output = vc./cosh(uc).^2;
        end
        % get g
        function [gb,gt] = get_g(self, vb, ub, vt, ut)
            gb = self.c1c3.*vb + self.k1k3.*ub - self.c3.*vt - self.k3.*ut;
            gt = - self.c3.*vb - self.k3.*ub + self.c2c3.*vt + self.k2k3.*ut;
            uc_lin = ub - ut;
            vc_lin = vb - vt;
            uc = Gohc(self, uc_lin);
            vc = dGohc(self, uc_lin, vc_lin);
            gb = gb - self.gamma .* ( self.c4.*vc + self.k4.*uc );
        end
        % time domain solver
        function [vb, ub, p] = solve_time_domain(self, f)
        %     Solve the cochlear model in time domain
        % 
        %     Parameters
        %     ----------
        %     f : ndarray
        %         Input signal [cm s^-2]
        % 
        %     Returns:
        %     --------
        %     vb : ndarray
        %         Basilar membrane (BM) velocity [cm s^-1]
        %     ub : ndarray
        %         Basilar membrane (BM) displacement [cm]
        %     p : ndarray
        %         Pressure difference between two chambers [barye]
        %         (1 [barye]= 0.1 [Pa])
            Ntime = round(length(f)/2);
            Nx = self.N;
            T = Ntime * self.dt;
            t2 = 0:self.dt/2:T;
            t = 0:self.dt:T;
            alpha2 = 4*self.rho*self.b/self.H/self.m1;
            vb = zeros(Ntime,Nx);
            ub = zeros(Ntime,Nx);
            vt = zeros(Ntime,Nx);
            ut = zeros(Ntime,Nx);
            p = zeros(Ntime,Nx);
            F = zeros(Nx,Nx);
            F(1,1) = -2 - alpha2*self.dx^2;
            F(1,2) = 2;
            for mm = 2:Nx-1
                F(mm,mm-1) = 1;
                F(mm,mm) = -2  - alpha2*self.dx^2;
                F(mm,mm+1) = 1;
            end
            F(end,end-1) = 1;
            F(end,end) = -2  - alpha2*self.dx^2;
            F = F./ self.dx^2;
            iF = inv(F);

            hw = waitbar(0);
            for ii = 1:Ntime-2
                waitbar(ii/(Ntime-1),hw);
                %%%%%  RK4 %%%%%%%%%

                % (ii)
                [gb, gt] = get_g(self, vb(ii,:), ub(ii,:), vt(ii,:), ut(ii,:));
                k = -alpha2*gb;
                k(1) = k(1) - f(ii*2) * 2/self.dx;

                %(iii)
                p(ii,:) = (iF*k')'; % <- this could be optimized (rtachi)

                %(iv)-(v)
                dvb1 = (p(ii,:)-gb)/self.m1;
                ub1 = ub(ii,:) + 0.5*self.dt*vb(ii,:);
                vb1 = vb(ii,:) + 0.5*self.dt*dvb1;
                dvt1 = -gt/self.m2;
                ut1 = ut(ii,:) + 0.5*self.dt*vt(ii,:);
                vt1 = vt(ii,:) + 0.5*self.dt*dvt1;

                % (ii)
                [gb, gt] = get_g(self, vb1, ub1, vt1, ut1);
                k = -alpha2*gb;
                k(1) = k(1) - f(ii*2+1) * 2/self.dx;

                %(iii)
                p1 = (iF*k')';

                %(iv)-(v)
                dvb2 = (p1-gb)/self.m1;
                ub2 = ub(ii,:) + 0.5*self.dt*vb1;
                vb2 = vb(ii,:) + 0.5*self.dt*dvb2;
                dvt2 = -gt/self.m2;
                ut2 = ut(ii,:) + 0.5*self.dt*vt1;
                vt2 = vt(ii,:) + 0.5*self.dt*dvt2;   

                % (ii)
                [gb, gt] = get_g(self,vb2, ub2, vt2, ut2);
                k = -alpha2*gb;
                k(1) = k(1) - f(ii*2+1) * 2/self.dx;

                %(iii)
                p2 = (iF*k')';

                %(iv)-(v)
                dvb3 = (p2-gb)/self.m1;
                ub3 = ub(ii,:) + self.dt*vb2; 
                vb3 = vb(ii,:) + self.dt*dvb3;
                dvt3 = -gt/self.m2;
                ut3 = ut(ii,:) + self.dt*vt2;
                vt3 = vt(ii,:) + self.dt*dvt3;  

                % (ii)
                [gb, gt] = get_g(self, vb3, ub3, vt3, ut3);
                k = -alpha2*gb;
                k(1) = -f(ii*2+2) * 2/self.dx;

                %(iii)
                p3 = (iF*k')';

                %(iv)-(v)
                dvb4 = (p3-gb)/self.m1;
                dvt4 = -gt/self.m2;
                ub(ii+1,:) = ub(ii,:) + self.dt/6*(vb(ii,:) + 2*vb1 + 2*vb2 + vb3);
                vb(ii+1,:) = vb(ii,:) + self.dt/6*(dvb1 + 2*dvb2 + 2*dvb3 + dvb4) ;
                ut(ii+1,:) = ut(ii,:) + self.dt/6*(vt(ii,:) + 2*vt1 + 2*vt2 + vt3);
                vt(ii+1,:) = vt(ii,:) + self.dt/6*(dvt1 + 2*dvt2 + 2*dvt3 + dvt4);
            end
            close(hw);
        end
    end
end
