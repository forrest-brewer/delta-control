function [psdx, freq] = psd_plot(x,Fs)
   
    % x = input signal 
    % Fs = sampling frequency

    N = length(x);
    xdft = fft(x);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(x):Fs/2;

    semilogx(freq,10*log10(psdx));
    ylim([-160 20]) %you may want to adjust the y limit here
    grid on
    title('PSD')
    xlabel('Frequency (Hz)')
    ylabel('Power/Frequency (dB/Hz)')
end
