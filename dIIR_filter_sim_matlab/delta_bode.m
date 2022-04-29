function [mag, phz] = delta_bode(A,B,C,D,f,ts)

q = size(C,1);
p = size(B,2);

fs = 1/ts;
mag = zeros(q,p,length(f));
phz = zeros(q,p,length(f));
delta = (exp(1i*2*pi*(f./fs))-1)./ts;

for i = 1:length(f)
    h = C*((delta(i)*eye(size(A))-A)\B) + D;
    mag(:,:,i) = abs(h);
    phz(:,:,i) = 180*atan2(imag(h),real(h))/pi;
end

end
