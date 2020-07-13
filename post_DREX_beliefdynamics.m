function delta = post_DREX_beliefdynamics(mdl)
% Usage: delta = post_DREX_beliefdynamics(mdl)
% 
% Post-processing of D-REX model output for summary of dynamics in beliefs.
% Calculates belief change at each time.
%
% ===INPUT===
%     mdl          output from run_DREX_model.m
%
% ===OUTPUT===
%     delta        belief change, JS divergence in context beliefs at each
%                   time before and after observing x_t
%
%
% Benjamin Skerritt-Davis
% bsd@jhu.edu
% 07/05/2020

B = mdl.context_beliefs;
[memory, ntime] = size(B);

delta = zeros(ntime,1);
try
for t = 1:ntime-1
    if memory <= t
        b1 = B(2:memory,t);
        b1(1) = b1(1)+B(1,t);
        b2 = B(1:memory-1,t+1);
        delta(t+1) = calc_JS_divergence(b1, b2);
    else
        delta(t+1) = calc_JS_divergence(B(1:t,t), B(1:t,t+1));
    end
end
catch
    keyboard;
end

end


function d = calc_JS_divergence(x, y)
if numel(x) > length(x(:));   error('x should be vector');  end
if numel(y) > length(y(:));   error('y should be vector');  end

x = reshape(x,[],1);
y = reshape(y,[],1);
z = 0.5*x + 0.5*y;

d1 = x'*safelog2(x./z);
d2 = y'*safelog2(y./z);

d = 0.5*d1 + 0.5*d2;
end

function out = safelog2(in)
out = log2(in);
out(isinf(out) | isnan(out)) = 0;

end
