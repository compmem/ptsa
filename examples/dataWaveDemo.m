
% load events
fprintf('Loading events...');
ev = loadEvents('/home1/per/eeg/free/CH012/events/events.mat');

% split out two conditions (recalled and not recalled)
rInd = inStruct(ev,'recalled==1');
nInd = inStruct(ev,'recalled==0');

% get power for the events for a range of freqs

% we leave the buffer on after getting the data, but pull it off
% in the call to tsPhasePow
freqs = [2:2:80];
chan = 27;
durationMS = 2500;
offsetMS = -500;
bufferMS = 1000;
resampledRate = 200;
filtFreq = [58.0,62.0];

% load the eeg data
rEEG = gete_ms(chan,ev(rInd),durationMS,offsetMS,bufferMS,filtFreq,'stop',4,resampledRate);
nEEG = gete_ms(chan,ev(nInd),durationMS,offsetMS,bufferMS,filtFreq,'stop',4,resampledRate);

% power for recalled events
rRes = getphasepow(chan,ev(rInd),durationMS,offsetMS,bufferMS,'freqs',freqs,...
                        'width',5,'filtfreq',filtFreq,'filttype','stop','filtorder',4,...
                        'resampledrate',resampledRate,'powonly');
nRes = getphasepow(chan,ev(nInd),durationMS,offsetMS,bufferMS,'freqs',freqs,...
                        'width',5,'filtfreq',filtFreq,'filttype','stop','filtorder',4,...
                        'resampledrate',resampledRate,'powonly');

% get mean power across events (axis=1)
rPow = squeeze(mean(log10(rRes),1));
nPow = squeeze(mean(log10(nRes),1));

% times
times = linspace(-500,2000,size(rEEG,2));

fprintf('Generating plots...')
fig = 0;

% erp
fig = fig + 1;
figure(fig);

plot(times,mean(rEEG,1),'r');
hold on
plot(times,mean(nEEG,1),'b');
hold off

legend('Recalled','Not Recalled')
xlabel('Time (ms)')
ylabel('Voltage (mV)')

keyboard


% power spectrum
fig=fig+1;
figure(fig)
plot(freqs,N.squeeze(N.mean(rPow,rPow.dim('event'))),'r')
plot(freqs,N.squeeze(N.mean(nPow,nPow.dim('event'))),'b')
legend('Recalled','Not Recalled')
xlabel('Frequency (Hz)')
ylabel('Power ($log_{10}(mV^2)$)')

% plot the diff in mean power
fia=fig+1;
figure(fig)
contourf(times,freqs,rPow-nPow)
colorbar()
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
%title('SME (diff in power) for channel %d' % (chan))



