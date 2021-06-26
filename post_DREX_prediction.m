function [Psi,X,Y] = post_DREX_prediction(featureidx, mdl_out, pred_pos)
% Evaluate predictive distribution from D-REX model output
%
% ===INPUT===
%   featureidx  index of feature to generate predictions from model output
%   mdl_out     output structure of D-REX model
%   pred_pos    vector of positions (in feature dimension) at which to evaluate predictive probability
%
% ===OUTPUT===
%   Psi         Predictive probability at each time evaluated at each position (dim: position x time)
%   X           mesh grid of horizontal positions for plotting Psi (dimensions: time)
%   Y           mesh grid of vertical positions for plotting Psi (dimension: feature)
%
%   NOTE: see 'display_DREX_output.m' for plotting example
% 


distribution = mdl_out.distribution;
if strcmp(distribution,'lognormal')
    pred_pos(pred_pos<=0) = [];
end
pp = mdl_out.prediction_params;
b = mdl_out.context_beliefs; % dim: hyp x time
ntime = length(pp);
nsample = length(pred_pos);
Psi = zeros(ntime, nsample);

for t = 1:ntime
    b_t = b(1:min(t,size(b,1)),t);
    pp_t = pp{t};
    
    for ib = 1:length(b_t)
        if b_t(ib)==0
            continue;
        end
        switch distribution
            case 'gaussian'
                Psi(t,:) = Psi(t,:) + b_t(ib)*studentpdf(pred_pos, pp_t.mu(ib,featureidx), pp_t.cov(ib,featureidx), pp_t.n(ib,featureidx));
            case 'lognormal'
                Psi(t,:) = Psi(t,:) + b_t(ib)*studentpdf(log(pred_pos), pp_t.mu(ib,featureidx), pp_t.cov(ib,featureidx), pp_t.n(ib,featureidx));
            case 'gmm'
                for ic = 1:pp_t.k{featureidx}(ib)
                    try
                    Psi(t,:) = Psi(t,:) + b_t(ib)*pp_t.pi{featureidx}(ib,ic)*studentpdf(pred_pos, pp_t.mu{featureidx}(ib,ic), pp_t.sigma{featureidx}(ib,ic), pp_t.n{featureidx}(ib,ic));
                    catch e
                        getReport(e)
                        keyboard;
                    end
                end
            otherwise
                error('post_DREX_prediction: distribution no supported')
        end
    end
    
end

[X,Y] = meshgrid(1:ntime, pred_pos); 
Psi = Psi'; % Flip for display display_DREX_output.m
end


function p = studentpdf(x, mu, var, n)
c = exp(gammaln(n/2 + 0.5) - gammaln(n/2)) .* (n.*pi.*var).^(-0.5);
p = c .* (1 + (1./(n.*var)).*(x-mu).^2).^(-(n+1)/2);
end
