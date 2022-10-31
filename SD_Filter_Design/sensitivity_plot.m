function sensitivity_plot(A,B,C,D,f,ts,S_mag,S_phz,q)

%sensitivity plot

[m_h,p_h] = delta_bode(A,B,C,D,f,ts);
m_h = squeeze(m_h);
p_h = squeeze(p_h);

up_m = m_h + (q'*S_mag)';
low_m = m_h - (q'*S_mag)';
low_m(low_m<0) = 0;

up_p = p_h + (q'*S_phz)';
low_p = p_h - (q'*S_phz)';

figure; subplot(2,1,1);
semilogx(f,20*log10(m_h),f,20*log10(up_m),'r--',f,20*log10(low_m),'g--');
xlabel('Frequency');
ylabel('Magnitude (dB)');
title('Bode Plot');
legend('Ideal Transfer Function','Upper Deviation Bound',...
    'Lower Deviation Bound');

subplot(2,1,2);
semilogx(f,p_h,f,up_p,'r--',f,low_p,'g--');
xlabel('Frequency');
ylabel('Phase');
legend('Ideal Transfer Function','Upper Deviation Bound',...
    'Lower Deviation Bound');

end