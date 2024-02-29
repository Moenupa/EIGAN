function DISP = rm_SACPZ(raw,zeros,poles,gain,samprate)
% remove instrument response from raw data into DISP
% the poles, zeros and gain values should get from SAC pole zero files (DISP)
% figure(101) shows RESP ampitude in VEL and DISP
% pylin.patty 2014.03

% should write one for RESP file, and normalized RESP amplitude at sensitive frequency and then multiply gain.
% pylin.patty 2014.04

% Sample rate is coming in in Hz

isfigure = 0;
raw = double(raw);
raw=detrend(raw);
% raw = flat_hanning_win(1:length(raw),raw,1,length(raw),50);
raw=cos_taper(raw);

N=length(raw);
T=N*(1./samprate);

% Setup frequency axis
if mod(N,2)
    faxis = [0:(N-1)/2,-(N-1)/2:-1]*(1/T);
else
    faxis = [0:N/2,-N/2+1:-1]*(1/T);
end

w = faxis.*2*pi;

% Introduce poles and zeros
resp = ones(size(w));
for ip = 1:length(poles)
    resp = resp./(i*w - poles(ip));
end
for ip = 1:length(zeros)
    resp = resp.*(i*w - zeros(ip));
end
resp = resp*gain;

if isfigure
% plotting ===
figure(101)
clf
subplot(2,1,2)
loglog(faxis,abs(resp));hold on;grid on;
hold on
loglog([1/samprate 1/samprate],[1e0 1e15],'-','color','r')
loglog([samprate/2 samprate/2],[1e0 1e15],'-','color','r')
title('DISP response');
subplot(2,1,1)
loglog(faxis,abs(resp)./faxis/2/pi);grid on;
hold on
loglog([1/samprate 1/samprate],[1e4 1e10],'-','color','r')
loglog([samprate/2 samprate/2],[1e4 1e10],'-','color','r')
title('VEL response');
% print(figure(101),'-dpsc','loglog.ps')

% figure(103)
% clf
% set(gcf,'position',[360   514   900   400]);
% hold on
% subplot(1,2,1)
% set(gca,'fontsize',18)
% semilogy(faxis,abs(resp),'rx');
% subplot(1,2,2)
% set(gca,'fontsize',18)
% 	plot(faxis,angle(resp),'rx');


% figure(102)
% clf
% plot(w/2.0/pi,abs(norm_trans),'bx');

end
% =============

lo_corner = 0.005;  % in Hz
npoles=5;
lo_w=2*pi*lo_corner;
hpfiltfrq=( ((w./lo_w).^(2*npoles))./(1+(w./lo_w).^(2*npoles)) );
%npoles = 5
%hi_corner = 15;
%hi_w=2*pi*hi_corner;
%hpfiltfrq=  hpfiltfrq - ( ((w./hi_w).^(2*npoles))./(1+(w./hi_w).^(2*npoles)) );

norm_trans=hpfiltfrq./resp;    % this is normalization transfer function
norm_trans(find(isnan(norm_trans))) = 0;




fftdata = fft(raw);
fftdata = fftdata(:).*norm_trans(:);
DISP = real(ifft(fftdata));
DISP = DISP-mean(DISP);

% disp('Removed response!')
return
