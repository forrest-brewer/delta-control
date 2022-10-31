clear
close all
clc

% Filter Specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256;
fb = 22050;
fs = OSR*2*fb;
ts = 1/fs;
f = logspace(0,log10(fb),2^10);

num_samples = 2^ceil(log2(1e6));
t_stop = ts*(num_samples-1);
t = 0:ts:t_stop;

amp = 0.5;
fsig = 100*fs/num_samples;
in = amp*sin(2*pi*fsig*t);
%in = amp*(2*rand(1,length(t))-1);

% Filter Design
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%lowpass filter
Rs = 60;
Wn = 2*pi*8000;
ftype = 'low';
N = 6;
[A,B,C,D] = cheby2(N,Rs,Wn,ftype,'s');
[Ad,Bd,Cd,Dd] = c2delta(A,B,C,D,ts);

% %cheby2 bandpass filter
% Rs = 60;
% Wn = 2*pi.*[300 3000];
% ftype = 'bandpass';
% N = 8;
% [z,p,k] = cheby2(N/2,Rs,Wn,ftype,'s');
% [A,B,C,D] = zp2ss(z,p,k);
% [T, A] = balance(A);
% B = T\B;
% C = C*T;
% [Ad,Bd,Cd,Dd] = c2delta(A,B,C,D,ts);


% Filter Implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Structural transformation of filter
[Ad_t,Bd_t,Cd_t,Dd_t,T0] = obsv_cst(Ad,Bd,Cd,Dd);
[num_t, den_t] = ss2tf(Ad_t,Bd_t,Cd_t,Dd_t);

% Scaling
[Ts, k] = dIIR_scaling(num_t,den_t,Ad,Bd,T0,f,ts);
K_inv = diag(k);
Ad_ts = Ts\Ad_t*Ts;
Bd_ts = Ts\Bd_t;
Cd_ts = Cd_t*Ts;
Dd_ts = Dd_t;
num_ts = [num_t(1) num_t(2:end)./(diag(Ts)')];
den_ts = [1 den_t(2:end)./(diag(Ts)')];

beta = num_ts;
alpha = den_ts;
bode(ss(A,B,C,D),tf(num_t,den_t));


% Bitwidth Optimization and Filter Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculation of sensitivity matrix
[S_mag, S_phz] = SD_dIIR_sensitivity(Ad,Bd,Cd,Dd,T0,Ts,f,ts);

% Calculation of quantization noise
[sig_nom, sig_2_x_sd, h1] = dDFIIt_noise_gain(Ad,Bd,Cd,Dd,K_inv,Ts,T0,f,ts);

% Bitwidth Optimization
SNR = 90;
sig_noise = 10^(-(SNR/10)); %-sig_nom;
p = .1*ones(1,length(S_mag));
s = sqrt(trapz(sig_2_x_sd,2)*(12*OSR));
q = bitwidth_opt(squeeze(S_mag),p,h1,sig_noise,s);

figure;loglog(f,20*log10(q'*S_mag),f,20*log10(p),'r');
10*log10(sqrt(2)/(q'*h1*q))

qs = log2(q);
qs(qs>0) = 0;
qs = ceil(abs(qs))';
bi = ceil(log2(max(abs(alpha),abs(beta)))) + 1;
bw = bi + qs + 1;
b_frac = qs;

% Sensitivity Plot
sensitivity_plot(Ad,Bd,Cd,Dd,f,ts,S_mag,S_phz,q);

% Simulink Model Bitwidth Parameters
shift = round(abs(log2(ts*k)));
bw_accum = 1 + ceil(abs(log2(ts))) + b_frac(2:end);
disp(['q = ' num2str(log2(q'))]);
disp(['Coefficient bitwidths = ' num2str(bw)]);
disp(['Bits of state = ' num2str(3*bw_accum(1) + sum(bw_accum(2:end)))]);

% sim('test');

switch N
    case 2 
        sim('Filter_Sim_2nd_Order')
    case 4 
        sim('Filter_Sim_4th_Order')
    case 6 
        sim('Filter_Sim_6th_Order')
    case 8 
        sim('Filter_Sim_8th_Order')
    case 12 
        sim('Filter_Sim_12th_Order')
    otherwise
end

% Plotting (PSD and Transfer Function Estimation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 0
    figure; hold on;
    nx = max(size(filter_output));
    na = 4;
    w = hanning(floor(nx/na));
    
    [H_mag, ~] = delta_bode(Ad,Bd,Cd,Dd,f,ts);
    semilogx(f,20*log10(abs(squeeze(H_mag)))); 
    
    [T_est1, ~] = tfestimate(sd1_output,filter_output,w,[],f,fs);
    semilogx(f,20*log10(abs(T_est1)));
    
    [T_est2, ~] = tfestimate(2*sd1_output1-1,2*filter_output1-1,w,[],f,fs);
    semilogx(f,20*log10(abs(T_est2)),'g');
       
    axis tight
    title('Transfer Function Magnitude Estimation');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    set(gca, 'XScale', 'log')
    legend('Ideal Transfer Function',...
        'Ideal \Sigma\Delta Filter',...
        'Fixed Point \Sigma\Delta Filter','Location','best');
    
    figure;
    [T_est3, ~] = tfestimate(sd1_output,state,w,[],f,fs);
    loglog(f,abs(T_est3)');
end


if 1
    %second order sigma delta PSD plot
    figure; hold on;
    N = length(sd_output1);
    xdft = fft(2*sd_output1-1);
    xdft = xdft(1:N/2+1);
    psdx = (1/(fs*N))*abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:fs/N:fs/2;
    
    semilogx(freq,10*log10(psdx),'b');
    
    %calculation of SNR
    fbin = freq(2)-freq(1);
    index = find(freq == fsig);
    sig_pwr = psdx(index)*fbin;
    freq(index) = [];
    psdx(index) = [];
    f = find(freq <= fb);
    noise_pwr = trapz(freq(f),psdx(f));
    snr = 10*log10(sig_pwr/noise_pwr);
    ENOB = (snr-1.76)/6.02;
    disp(['ENOB = ' num2str(ENOB)]);
end



