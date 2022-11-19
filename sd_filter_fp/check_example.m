clear
close all
clc

addpath('..\examples')

% Filter Specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256;                          %oversample ratio
fb = 22050;                         %nyquist
fs = OSR*2*fb;                      %sampling frequency
ts = 1/fs;                          %sampling period
f = logspace(0,log10(fb),2^10);

% Filter Input Signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
amp = 0.5;
t_stop = 2;
t = 0:ts:t_stop;
in = amp*chirp(t,0,1,fb/2);

fprintf('min input: %f\n',min(in));
fprintf('max input: %f\n',max(in));

sd_filter = matfile('../examples/cheby2_bandpass.mat');
% sd_filter = matfile('../examples/cheby2_1k_highpass.mat');
k         = sd_filter.k;
beta      = sd_filter.beta;
alpha     = sd_filter.alpha;
q         = sd_filter.q;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qs = log2(q);
qs(qs>0) = 0;
qs = ceil(abs(qs))';
bi = ceil(log2(max(abs(alpha),abs(beta)))) + 1;
bw = bi + qs + 1;
b_frac = qs;

% % Sensitivity Plot
% sensitivity_plot(Ad,Bd,Cd,Dd,f,ts,S_mag,S_phz,q);

% Simulink Model Bitwidth Parameters
shift = round(abs(log2(ts*k)));
bw_accum = 1 + ceil(abs(log2(ts))) + b_frac;
disp(['q = ' num2str(log2(q'))]);
disp(['Coefficient bitwidths = ' num2str(bw)]);



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simin.signals.values = in';
simin.signals.values = fi(in, 1, 16, 7)';
simin.time = t';

k_ts = k .* ts;
k_ts(end+1) = 0;
shift(end+1) = 1;

% bw(2) = bw(2) + 1
% bw(4) = bw(4) + 1

for i = 1:length(alpha)
  bitwidths(i).accum = fi(0       , 1, bw_accum(i)    , b_frac(i));
  bitwidths(i).alpha = fi(alpha(i), 1, bw(i)       + 0, qs(i)    );
  bitwidths(i).beta  = fi(beta(i) , 1, bw(i)       + 0, qs(i)    );
  bitwidths(i).k_ts  = fi(k_ts(i) , 1, shift(i)    + 1, shift(i) );
end

% for i = 1:length(alpha)
  % bitwidths(i).accum = 0       ;
  % bitwidths(i).alpha = alpha(i);
  % bitwidths(i).beta  = beta(i) ;
  % bitwidths(i).k_ts  = k_ts(i) ;
% end

sd_mod_amp = 1;
sd_mod_amp = fi(sd_mod_amp, 1, 32, 15);

% Simulate Simulink Filter Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_out = sim('sd_filter_tb');

filter_output = sim_out.filter_output.signals.values;

disp(sim_out.filter_output.signals.dimensions)

psd_plot(filter_output,fs);



