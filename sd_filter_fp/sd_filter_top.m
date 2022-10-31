clear
close all
clc

addpath('..\dIIR_filter_sim_matlab')

% Filter Specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256;                          %oversample ratio
fb = 22050;                         %nyquist
fs = OSR*2*fb;                      %sampling frequency
ts = 1/fs;                          %sampling period
f = logspace(0,log10(fb),2^10);

% Filter Input Signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amp = 0.5;
% t_stop = 2;
% t = 0:ts:t_stop;
% in = amp*chirp(t,0,1,fb/2);

num_samples = 1e7;    %number of simulation samples
t_stop = ts*(num_samples-1);
t = 0:ts:t_stop;
amp = 1.2;
in = amp*(2*rand(1,length(t))-1); %white noise input
% [in,d] = lowpass(in,fb,fs,ImpulseResponse="fir",Steepness=0.95);
[in,d] = lowpass(in,fb,fs);


fprintf('min input: %f\n',min(in));
fprintf('max input: %f\n',max(in));

% Filter Design
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % lowpass filter
% Rs = 60;
% Wn = 2*pi*2000;
% ftype = 'low';
% % N = 6;
% N = 4;
% [A,B,C,D] = cheby2(N,Rs,Wn,ftype,'s');
% [Ad,Bd,Cd,Dd] = c2delta(A,B,C,D,ts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%cheby2 bandpass filter
Rs = 60;
Wn = 2*pi.*[300 3000];
ftype = 'bandpass';
N = 4;
[z,p,k] = cheby2(N/2,Rs,Wn,ftype,'s');
[A,B,C,D] = zp2ss(z,p,k);
[T, A] = balance(A);
B = T\B;
C = C*T;

%convert to delta domain
[Ad,Bd,Cd,Dd] = c2delta(A,B,C,D,ts);

% Filter Implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Structural transformation of filter
[Ad_t,Bd_t,Cd_t,Dd_t,T0] = obsv_cst(Ad,Bd,Cd,Dd);
[num_t, den_t] = ss2tf(Ad_t,Bd_t,Cd_t,Dd_t);

% Scaling
[Ts, k] = dIIR_scaling(Ad,Bd,T0,f,ts);
K_inv = diag(k);
Ad_ts = Ts\Ad_t*Ts;
Bd_ts = Ts\Bd_t;
Cd_ts = Cd_t*Ts;
Dd_ts = Dd_t;
num_ts = [num_t(1) num_t(2:end)./(diag(Ts)')];
den_ts = [1 den_t(2:end)./(diag(Ts)')];

beta = num_ts;
alpha = den_ts;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(alpha);
disp(beta );
disp(k    );

log2(alpha)
log2(beta )
log2(k    )


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
% bw_accum = 1 + ceil(abs(log2(ts))) + b_frac(2:end);
bw_accum = 1 + ceil(abs(log2(ts))) + b_frac;
disp(['q = ' num2str(log2(q'))]);
disp(['Coefficient bitwidths = ' num2str(bw)]);
% disp(['Bits of state = ' num2str(3*bw_accum(1) + sum(bw_accum(2:end)))]);


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simin.signals.values = in';
simin.signals.values = fi(in, 1, 16, 7)';
simin.time = t';

k_ts = k .* ts;
k_ts(end+1) = 0;
shift(end+1) = 1;

for i = 1:length(alpha)
  bitwidths(i).accum = fi(0       , 1, bw_accum(i) , b_frac(i));
  bitwidths(i).alpha = fi(alpha(i), 1, bw(i)       , qs(i)    );
  bitwidths(i).beta  = fi(beta(i) , 1, bw(i)       , qs(i)    );
  bitwidths(i).k_ts  = fi(k_ts(i) , 1, shift(i) + 1, shift(i) );
end

sd_mod_amp = 1
sd_mod_amp = fi(sd_mod_amp, 1, 16, 7)

% Simulate Simulink Filter Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_out = sim('sd_filter_tb');

filter_output = sim_out.filter_output.signals.values;
% filter_state = sim_out.filter_state.signals.values;

disp(sim_out.filter_output.signals.dimensions)
% disp(sim_out.filter_state.signals.dimensions)

psd_plot(filter_output,fs);


% filter_output = []
% filter_state = []
% in = []
% t = []
% w = []
% sim_out = []
% ans = []
% save('all.mat')

