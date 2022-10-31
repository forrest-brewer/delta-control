function psd_plot(x,fs,na)

%     N = length(x);
%     xdft = fft(x);
%     xdft = xdft(1:N/2+1);
%     psdx = (1/(fs*N)) * abs(xdft).^2;
%     psdx(2:end-1) = 2*psdx(2:end-1);
%     freq = 0:fs/length(x):fs/2;
nx = max(size(x));
w = hanning(floor(nx/na));
[psdx,freq]=pwelch(x,w,0,[],fs);

semilogx(freq,10*log10(psdx));
grid on
title('PSD')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

end