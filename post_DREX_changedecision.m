function out = post_DREX_changedecision(mdl, thresh)
% Usage: out = post_DREX_changedecision(mdl, thresh)
% 
% Post-processing of D-REX model output for change detection task where
% single change may or may not have occurred in observed sequence.
%
% Calculates change signal and change decision from context beliefs.
%
% ===INPUT===
%     mdl                 output from run_DREX_model.m
%     thresh              threshold for change probability to detect first change, scalar [0,1] (default=nan)
%
% ===OUTPUT===
%     out (struct)
%       changesignal        probability at least one change occured since beginning of input sequence (dimord: time x 1)
%       decision            boolean, whether changesignal exceeded decision threshold (see params.thresh)
%       changepoint         time when changesignal crossed decision threhsold
%
% v2
% Benjamin Skerritt-Davis
% bsd@jhu.edu
% 12/27/2019

if nargin < 2
    thresh = nan;
end

[memory, ntime] = size(mdl.context_beliefs);
B = mdl.context_beliefs;

% Change probability: at least one change in observed sequence
changeprobability = zeros(ntime,1);
for t = 1:ntime
    if t > 1 && t > memory
        changeprobability(t) = max(changeprobability(t-1), 1 - B(1,t)); % after t > memory, keep max
    else
        changeprobability(t) = 1 - B(1,t);
    end
end

% Change decision: change prob exceeded threshold                   
if isnan(thresh)                
    decision = nan;
    changept = nan;
else 
    decision = changeprobability(end) > thresh; % 1 = change, 0 = no change
    if decision
        changept = find(changeprobability < thresh, 1, 'last')+1;
    else
        changept = nan;
    end
end

out.changeprobability = changeprobability;
out.decision = decision;
out.changepoint = changept;

end