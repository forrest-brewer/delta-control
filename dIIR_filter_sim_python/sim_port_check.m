clear
close all
clc
addpath('..\dIIR_filter_sim_matlab')

% Filter Specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256;                          %oversample ratio
fb  = 22050;                         %nyquist
fs  = OSR*2*fb;                      %sampling frequency
ts  = 1/fs;                          %sampling period
f   = logspace(0,log10(fb),2^10);

num_samples = 1e7;    %number of simulation samples
t_stop = ts*(num_samples-1);
t = 0:ts:t_stop;

% Filter Input Signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
amp = 0.5;
in  = amp*(2*rand(1,length(t))-1); %white noise input

% load filter design from python
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
example = matfile('./py_to_mat.mat');
Ad    = example.Ad;
Bd    = example.Bd;
Cd    = example.Cd;
Dd    = example.Dd;
k     = example.k;
beta  = example.beta ;
alpha = example.alpha;

% Simulate Simulink Filter Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_out = sim('filter_model');

filter_output = sim_out.filter_output.signals.values;
filter_state = sim_out.filter_state.signals.values;

% Plotting (Transfer Function Estimation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; hold on;
nx = max(size(filter_output));
na = 4;
w = hanning(floor(nx/na));

[H_mag, ~] = delta_bode(Ad,Bd,Cd,Dd,f,ts);
semilogx(f,20*log10(abs(squeeze(H_mag))),'linewidth',3);

[T_est1, ~] = tfestimate(in,filter_output,w,[],f,fs);
semilogx(f,20*log10(abs(T_est1)),'linewidth',3);

axis tight
title('Transfer Function Magnitude Estimation');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
set(gca, 'XScale', 'log');
legend('Ideal Transfer Function',...
    'Ideal \delta Filter Simulation', 'location', 'best');

% Plotting (State Variables)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
[T_est2, ~] = tfestimate(in,filter_state,w,[],f,fs);
loglog(f,abs(T_est2)','linewidth',3);

axis tight
title('State Variable Bode Plot');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
set(gca, 'XScale', 'log');
