function [env, time] = extract_envelope(wavs, fs, windowsize, stepsize)
%EXTRACT_ENVELOPE Summary of this function goes here
%   Detailed explanation goes here


if nargin < 3
    windowsize = 10;
end
if nargin < 4
    stepsize = 5;
end

if ~iscell(wavs) || any(cellfun(@numel,wavs) - cellfun(@length,wavs))
    error('Input ''wavs'' should be cell array with mono acoustic signal (1-D array) in each cell')
end

if isempty(which('wav2aud'))
    error('NSL toolbox not installed. Please download at http://nsl.isr.umd.edu/downloads.html and add to MATLAB path');
end


% Extract RMS envelope from acoustic signal in each cell
env = cell(size(wavs));
time = cell(size(wavs));
for i = 1:length(wavs)
    
    nframe = floor((length(wavs{1}) - round(windowsize/1000*fs{i}))/round(stepsize/1000*fs{i})) + 1;
    framelength = round(windowsize/1000*fs{i});
    steplength = round(stepsize/1000*fs{i});
    env{i} = zeros(nframe,1);
    for f = 1:nframe
        env{i}(f) = rms(wavs{i}((1:framelength)+(f-1)*steplength));
    end
    time{i} = (0:nframe-1)'*stepsize/1000; % sec
end


end

