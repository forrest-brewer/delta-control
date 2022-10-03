clear
close all
clc

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

% Simulate Simulink Filter Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_out = sim('sd_2nd_mod_tb');
filter_output = sim_out.filter_output.signals.values;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psd_plot(filter_output,fs);

% [psdx_out, freq_out] = psd_plot(filter_output,fs);
% [psdx_in, freq_in] = psd_plot(in,fs);
% loglog(freq_in, psdx_out ./ psdx_in')


