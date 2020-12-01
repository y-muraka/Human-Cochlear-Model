% demo CM1D plot
clear

% parameters
Nx = 300;
g = 1;
gamma = ones(1,Nx)*g;
fs_model = 200e3;

% Initial setup
cm = CochlearModel1D(Nx, gamma); 

% three wavefiles (read only initial 2400 samples)
readrng = [1 2400];
Lps = 36; % 36 dB input
fnames = {'250Hz.wav','1000Hz.wav','4000Hz.wav'};
vbs = cell(3,1);
ubs = cell(3,1);
ps = cell(3,1);
%     vb : Basilar membrane (BM) velocity [cm s^-1]
%     ub : Basilar membrane (BM) displacement [cm]
%     p : Pressure difference between two chambers [barye]
%         (1 [barye]= 0.1 [Pa])
for f = 1:3
    [wav,fs] = audioread(fnames{f},readrng);
    dt = 1/fs;
    wav = wav/max(abs(wav));
    res = resample(wav,fs_model,fs);
    signal = zeros(length(res),1);
    signal(2:end-1) = (res(3:end)-res(1:end-2))/2/dt;
    multi = 20e-5*10^(Lps/20.0);
    signal = multi*signal;
    [vb, ub, p] = cm.solve_time_domain( signal ); % Solve
    vbs(f) = {vb};
    ubs(f) = {ub};
    ps(f) = {p};
end

%% movie for BM displacement
nframes = size(vb,1);
moviestep = 30;
for n=1:moviestep:nframes
    for f=1:3
        subplot(3,1,f);
        plot(cm.x*10,ubs{f}(n,:)*10*1000*1000);
        xlabel('Distance from the stapes [mm]');
        ylabel('BM displacement [nm]');
        ylim([-70 70]);
    end
    subplot(3,1,1);
    title(sprintf('(%d/%d) %0.3f ms',n,nframes,n/fs_model*1000));
    drawnow;
end

