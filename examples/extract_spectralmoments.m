function [centroid, spread, skw, krt, time] = extract_spectralmoments(wavs, fs, timestep, useMedian)


if nargin < 3
    timestep = 25;
end
if nargin < 4
    useMedian = false;
end

if ~iscell(wavs) || any(cellfun(@numel,wavs) - cellfun(@length,wavs))
    error('Input ''wavs'' should be cell array with mono acoustic signal (1-D array) in each cell')
end

if isempty(which('wav2aud'))
    error('NSL toolbox not installed. Please download at http://nsl.isr.umd.edu/downloads.html and add to MATLAB path');
end


% Resample to 16000 Hz
for i = 1:length(wavs)
    if fs{i} ~= 16000
        wavs{i} = resample(wavs{i}, 16000, fs{i});
    end
    fs{i} = 16000;
end


load('aud24.mat','COCHBA');

%%%%%%% Cochlear filterbank parameters %%%%%%%%%%%
paras = [timestep 32 -2 log2(fs{1}/16000)];
cf = cochfil(1:129,paras(4)); 	% Center frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Extract pitch from acoustic signal in each cell
centroid = cell(size(wavs));
spread = cell(size(wavs));
skw = cell(size(wavs));
krt = cell(size(wavs));
for i = 1:length(wavs)
    
    % Get auditory spectrogram
    aud = wav2aud(unitseq(wavs{i}),paras)';
    norm_aud = aud ./ repmat(sum(aud,1),size(aud,1),1);
   
    % Centroid
    if useMedian
        csaud = cumsum(norm_aud);
        centroid{i} = zeros(1, size(aud,2));
        for t = 1:size(aud,2)
            centroid{i}(t) = cf(find(csaud(:,t) > 0.5,1));
        end
    else
        centroid{i} = sum(repmat(cf(1:size(aud,1))',1,size(aud,2)).* norm_aud,1);
    end
    
    % Spread
    spread{i} = sqrt(sum((repmat(cf(1:size(aud,1))',1,size(aud,2)) - repmat(centroid{i}, size(aud,1),1)).^2 .* norm_aud,1));
    
    % Skew
    skw{i} = sum((repmat(cf(1:size(aud,1))',1,size(aud,2)) - repmat(centroid{i}, size(aud,1),1)).^3 .* norm_aud,1) ./ (spread{i}.^3);
    
    % Kurtosis
    krt{i} = sum((repmat(cf(1:size(aud,1))',1,size(aud,2)) - repmat(centroid{i}, size(aud,1),1)).^4 .* norm_aud,1) ./ (spread{i}.^4);
    
end

time = (1:size(aud,2))*timestep/1000;

end