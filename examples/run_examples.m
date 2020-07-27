%{
Examples using D-REX model to generate predictions along different
dimensions from real-world audio clips.

External requirements: 
- NSL toolbox: http://nsl.isr.umd.edu/downloads.html
- MATLAB-MIDI toolbox: https://github.com/kts/matlab-midi 
- Audio source files: Audio S1-S8

Publication: Skerritt-Davis, B & Elhilali, M. (under review). Computational
framework for predictive processing of complex sound sequences.

%}

addpath('..'); % Add path to main directory containing D-REX model code (run_DREX_model.m)

%% Example 1
[x,fs] = audioread('AudioS1.mp3');
windur = 0.05; % sec
winsize = round(windur*fs);

xwinL = reshape(x(1:floor(length(x)/winsize)*winsize,2),winsize,[]);
xwinR = reshape(x(1:floor(length(x)/winsize)*winsize,1),winsize,[]);
x = zeros(size(xwinL,2),1);

for i = 1:length(x)
    xx = xwinL(:,i)+1e-8*randn(size(xwinL(:,i))); % add epsilon noise floor so dB-RMS >0
    yy = xwinR(:,i)+1e-8*randn(size(xwinR(:,i)));
    x(i) = 20*log10(rms(xx)/rms(yy));
end

ftsz = 14;

figure(1); clf;
params = [];
params.distribution = 'gaussian';
params.D = 1;
params.hazard = 0.01;
params.prior = estimate_suffstat(std(x)*randn(1000,1)+mean(x),params);
params.output_full_prediction = true;


out = run_DREX_model(x,params);
display_DREX_output(out,x);

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'Xtick',20:20:120,'xticklabel',(20:20:120)*windur)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'Xtick',20:20:120,'xticklabel',(20:20:120)*windur)
set(gca,'Ytick',20:20:120,'yticklabel',{'','2','','4','','6'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
ylim([10 18]);
set(gca,'Xtick',20:20:120,'xticklabel',(20:20:120)*windur)
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 


%% Example 2
figure(2); clf;
params = [];
params.distribution = 'gaussian';
params.D = 2;
params.hazard = 0.001;
params.prior = estimate_suffstat(std(x)*randn(1000,1)+mean(x),params);

out = run_DREX_model(x,params);
display_DREX_output(out,x)
subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'XTick',20:20:120,'xticklabel',(20:20:120)*windur)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'XTick',20:20:120,'xticklabel',(20:20:120)*windur)
set(gca,'Ytick',20:20:120,'yticklabel',{'','2','','4','','6'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
ylim([10 18])
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 
set(gca,'XTick',20:20:120,'xticklabel',(20:20:120)*windur)

%% Example 3
mid = readmidi('AudioS2.mid');
noteinfo = midiInfo(mid,0);
pitch = noteinfo(:,3);

x = pitch(1:140);

figure(3); clf;
params = [];
params.distribution = 'gaussian';
params.D = 10;
params.hazard = 0.001;
params.prior = estimate_suffstat(std(x)*randn(1000,1)+mean(x),params);
params.predscale = 1;
params.output_full_prediction = true;
 

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'Xtick',16:16:140,'xticklabel',(16:16:140)*0.25)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 
set(gca,'YTick',45:10:65,'YTickLabel', round(440*2.^(((45:10:65)-69)/12)) );

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'Xtick',16:16:140,'xticklabel',(16:16:140)*0.25)
set(gca,'Ytick',16:16:140,'yticklabel',{'','8','','16','','24','','32'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
ylim([0 15]);
set(gca,'Xtick',16:16:140,'xticklabel',(16:16:140)*0.25)
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 

%% Example 4
mid = readmidi('AudioS3.mid');
noteinfo = midiInfo(mid,0);
pitch = noteinfo(:,3);

x = pitch(100:300);

figure(4);
params = [];
params.distribution = 'gmm';
params.D = 1;
params.hazard = 0.001;
params.max_ncomp = 5;
params.beta = 0.05;
params.prior = estimate_suffstat(std(x)*randn(1000,1)+mean(x),params);
params.prior.sigma{1} = 1;
params.predscale = 1;

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'Xtick',32:32:200,'Xticklabel',0.1875*(32:32:200))
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 
set(gca,'YTick',56:9:74,'YTickLabel',round(440*2.^(((56:9:74)-69)/12)));
caxis([0 0.2])

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'Xtick',32:32:200,'Xticklabel',0.1875*(32:32:200))
set(gca,'Ytick',32:32:200,'yticklabel',{'','12','','24','','36'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
set(gca,'Xtick',32:32:200,'Xticklabel',0.1875*(32:32:200))
ylim([2 10]);
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 

%% Example 5
figure(5); clf;
[wav,fs] = audioread('AudioS4.mp3');
wav = resample(wav,16000, fs);
fs = 16000;
in = {wav};
[~,spr,~,~,~] = extract_spectralmoments(in,{fs},50);
x = spr{1}';
x(end) = [];

params = [];
params.distribution = 'gaussian';
params.D = 1;
params.hazard = 0.01;
params.prior = estimate_suffstat(spr{1}',params);
params.predscale = .01;
params.output_full_prediction = true;

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'XTick',20:20:100, 'XTickLabel',(20:20:100)*0.05)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'XTick',20:20:100, 'XTickLabel',(20:20:100)*0.05)
set(gca,'yticklabel',{'','2','','4',''},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
set(gca,'XTick',20:20:100, 'XTickLabel',(20:20:100)*0.05)
ylim([12 20]);
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 

%% Example 6
[wav,fs] = audioread('AudioS5.mp3');
wav = mean(wav,2);
wav = resample(wav,16000, fs);
fs = 16000;
in = {wav};
[c,~,~,~,~] = extract_spectralmoments(in,{fs},50);
x = c{1}';
x(end) = [];


figure(6); clf;
params = [];
params.distribution = 'gaussian';
params.D = 2;
params.hazard = 0.01;
params.prior = estimate_suffstat(x,params);
params.predscale = .01;
params.output_full_prediction = true;

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'XTick',20:20:160,'xticklabel',(20:20:160)*0.05)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 
ylim([400 1600])

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'XTick',20:20:160,'xticklabel',(20:20:160)*0.05)
set(gca,'YTick',20:20:160,'yticklabel',{'','2','','4','','6','','8'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
set(gca,'XTick',20:20:160,'xticklabel',(20:20:160)*0.05)
ylim([12 24]);
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 



%% Example 7
[wav,fs] = audioread('AudioS6.mp3');
wav = mean(wav,2);
wav = resample(wav,16000, fs);
fs = 16000;
in = {wav};
[env,time] = extract_envelope(in, {fs}, 100, 50);
x = env{1};
time = time{1};

figure(7); clf;
params = [];
params.distribution = 'lognormal';
params.D = 1;
params.hazard = 0.001;
params.prior = estimate_suffstat(x,params);
params.predscale = .001;
params.output_full_prediction = true;

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])

set(gca,'Xtick',20:20:120,'xticklabel',(20:20:120)*0.05)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'Xtick',20:20:120,'xticklabel',(20:20:120)*0.05)
set(gca,'ytick',20:20:120,'yticklabel',{'','2','','4','','6'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
set(gca,'Xtick',20:20:120,'xticklabel',(20:20:120)*0.05)
ylim([8 24]);
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 


%% Example 8
[wav,fs] = audioread('AudioS7.mp3');
wav = mean(wav,2);
wav = resample(wav,16000, fs);
fs = 16000;
in = {wav};
env = extract_envelope(in, {fs}, 200, 50);
x = env{1};

figure(8); clf;
params = [];
params.distribution = 'lognormal';
params.D = 2;
params.hazard = 0.01;
params.prior = estimate_suffstat(x,params);
params.predscale = .01;
params.output_full_prediction = true;

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'XTick',20:20:180,'XTicklabel',(20:20:180)*0.05)
set(gca,'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 
% set(gca,'YTick',56:9:74,'YTickLabel',round(440*2.^(((56:9:74)-69)/12)));

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'XTick',20:20:180,'XTicklabel',(20:20:180)*0.05)
set(gca,'YTick',20:20:180,'yticklabel',{'','','3','','','6','','','9'},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
set(gca,'XTick',20:20:180,'XTicklabel',(20:20:180)*0.05)
ylim([4 18]);
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 



%% Example 9
[wav,fs] = audioread('AudioS8.mp3');
wav = mean(wav,2);
wav = resample(wav,16000, fs);
fs = 16000;
in = {wav};

paras = [16 16 -2 0];
cf = cochfil(1:129,paras(4)); 	% Center frequencies
aud = wav2aud(unitseq(in{1}),paras)';
time = (1:size(aud,2))*paras(1)/1000;

ramp = (1:size(aud,1))';
ramp = ramp / sum(ramp); % normalize
aud = aud.*repmat(ramp,1,size(aud,2));

peakfxn = mean(aud(79:end,:)).^2; % sum >1760 Hz squared
peakfxn = peakfxn/max(peakfxn);

% extract peaks
peakwindow = 4; % samples (4=64 ms with 16 ms/sample)
peakthresh = median(peakfxn);
peaks = [];
i = 1;

while i+peakwindow <= length(peakfxn)
    [peakheight,maxwinidx] = max(peakfxn(i:i+peakwindow));
    if ismember(maxwinidx, [1 peakwindow+1]) || peakheight < peakthresh
        i = i+1;
    else
        peaks = [peaks i-1+maxwinidx];
        i = i+maxwinidx;
    end
end

x = zeros(length(peakfxn),1);
x(peaks) = 1;

figure(9); clf;
params = [];
params.distribution = 'poisson';
params.hazard = 0.01;
params.D = 25;
params.prior = estimate_suffstat({x},params);

out = run_DREX_model(x,params);
display_DREX_output(out,x)

subplot(311); title([]); xlabel([]); ylabel([])
set(gca,'XTick',2*63:2*63:900,'Xticklabel',2:2:14)
set(gca,'xticklabel',[],'ytick',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.65 0.775 0.2157]); 

subplot(312); title([]); xlabel([]); ylabel([]); box on; grid on; 
set(gca,'XTick',2*63:2*63:900,'Xticklabel',2:2:14)
set(gca,'yticklabel',{'','4','','8','','12',''},'xticklabel',[],'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.4 0.775 0.2157]); 

subplot(313); title([]); ylabel([]); xlabel([]);
set(gca,'XTick',2*63:2*63:900,'Xticklabel',2:2:14)
ylim([11 14]);
set(gca,'FontName','Arial','FontSize',ftsz,'Position',[0.13 0.15 0.775 0.2157],'ytick',ylim); 


