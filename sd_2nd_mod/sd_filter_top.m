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
amp = 0.5;
t_stop = 2;
t = 0:ts:t_stop;
in = amp*chirp(t,0,1,fb/2);

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
disp(beta);
disp(k);

% Simulate Simulink Filter Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sim_out = sim('sd_filter_tb');

filter_output = sim_out.filter_output.signals.values;
filter_state = sim_out.filter_state.signals.values;

disp(sim_out.filter_output.signals.dimensions)
disp(sim_out.filter_state.signals.dimensions)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% psd_plot(filter_state(:, 1),fs);
fprintf('min filter_state(1): %f\n', min(filter_state(:, 1)));
fprintf('max filter_state(1): %f\n', max(filter_state(:, 1)));

fprintf('min filter_state(2): %f\n', min(filter_state(:, 2)));
fprintf('max filter_state(2): %f\n', max(filter_state(:, 2)));

fprintf('min filter_state(3): %f\n', min(filter_state(:, 3)));
fprintf('max filter_state(3): %f\n', max(filter_state(:, 3)));

fprintf('min filter_state(4): %f\n', min(filter_state(:, 4)));
fprintf('max filter_state(4): %f\n', max(filter_state(:, 4)));

psd_plot(filter_output,fs);

