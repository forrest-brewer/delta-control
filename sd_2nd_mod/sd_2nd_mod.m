% clear
% close all
% clc

% addpath('..\dIIR_filter_sim_matlab')

% Filter Specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OSR = 256;                          %oversample ratio
fb = 22050;                         %nyquist
fs = OSR*2*fb;                      %sampling frequency
ts = 1/fs;                          %sampling period
f = logspace(0,log10(fb),2^10);

% num_samples = 2^ceil(log2(1e6));
% num_samples = 1e7;    %number of simulation samples
num_samples = 1e3;    %number of simulation samples
t_stop = ts*(num_samples-1);
t = 0:ts:t_stop;

% Filter Input Signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
amp = 0.5;
fsig = 100*fs/num_samples;
in = amp*sin(2*pi*fsig*t);
% in = amp*(2*rand(1,length(t))-1); %white noise input

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulate Simulink Filter Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_out = sim('sd_2nd_mod_tb');

filter_output = sim_out.filter_output.signals.values;
filter_state = sim_out.filter_state.signals.values;

