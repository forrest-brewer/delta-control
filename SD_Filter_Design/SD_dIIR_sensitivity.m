function [S_mag, S_phz] = SD_dIIR_sensitivity(A,B,C,D,T0,Ts,f,ts)

%Sensitivity due to feedthrough coefficient
S_beta0 = ss(A,A*T0*[1 zeros(1,size(A,2)-1)]',C,1);

%Sensitivity due to numerator coefficients
S_beta = ss(A',C',Ts'*T0',0);

%Sensitivity due to denominator coefficients
H = ss(A,B,C,D);
S_alpha = H*S_beta;
S_bsys = [S_beta0; S_beta];
S_asys = [0; S_alpha];
[S_bmag,S_bphz] = delta_bode(S_bsys.a,S_bsys.b,S_bsys.c,S_bsys.d,f,ts);
[S_amag,S_aphz] = delta_bode(S_asys.a,S_asys.b,S_asys.c,S_asys.d,f,ts);
S_mag = squeeze(S_bmag + S_amag);
S_phz = squeeze(S_bphz + S_aphz);

end