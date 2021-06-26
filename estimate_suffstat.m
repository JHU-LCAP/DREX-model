function suffstat = estimate_suffstat(xs, params)
% Usage: suffstat = estimate_suffstat(xs, D)
% 
% Estimates the sufficient statistics of the input, assuming a D-variate 
% gaussian (where D determines the temporal dependence).
% 
% ===INPUT===
%   xs          multi-trial, multi-feature input sequence in 3D array (time x trial x feature) -OR- cell array, {trial}(time,feature)
%   params      parameters for D-REX, for example:
%                   distribution    'gaussian', 'lognormal', 'gmm', 'poisson'
%                   D               temporal dependence (size of covariance structure)
%                   
% 
%
% ===OUTPUT===
%   suffstat    structure with sufficient statistics by feature
% 
% 
% * Suffstat structure fields
% Each field is a cell array with a cell for each feature. In each cell:
%     mu{f}       prior mean (size: D x 1)
%     ss{f}       prior sum of squares (size: D x D)
%     n{f}        prior observation count (size: 1 x 1)
%         
%
% NOTE: If covariance estimate has a negative eigenvalue (i.e., not positive-definite), regularization is added
%
%
% v2
% Benjamin Skerritt-Davis
% bsd@jhu.edu
% 6/25/2019

if ~isstruct(params); error('Usage: suffstat=estimate_suffstat(xs, params)'); end
if ~isfield(params,'D'), D = 1; else D = params.D; end
if ~isfield(params,'distribution'), dist = 'gaussian'; else dist = params.distribution; end

switch dist
    case 'gmm'
        if ~isfield(params,'max_ncomp'), max_ncomp = 10; else max_ncomp = params.max_ncomp; end
        D=1;
    case 'poisson'
        if ~isfield(params,'D'), D = 50; else D = params.D; end
end



if ~iscell(xs)
    old_xs = xs; % 
    xs = cell(size(old_xs,2),1);
    for t = 1:size(old_xs,2)
        xs{t} = squeeze(old_xs(:,t,:));
    end
end

ninput = length(xs);
nfeature = size(xs{1},2);

switch dist
    case 'gaussian'
        suffstat = [];
        suffstat.mu = cell(nfeature,1); % mu: mean
        suffstat.ss = cell(nfeature,1); % ss: sum of squared deviation from mean, "scatter matrix"
        suffstat.n = cell(nfeature,1);  % n: count
        
        for f = 1:nfeature
            mu = zeros(ninput,1);
            ss = zeros(ninput,D);
            n = zeros(ninput,1);
            for t = 1:ninput
                mu(t) = mean(xs{t}(:,f),1,'omitnan');
                tmpss = zeros(1,D);
                n(t) = size(xs{t},1);
                for d = 1:D
                    x = xs{t}(:,f);
                    tmpss(d) = mean((x(1:end-d+1)-mu(t)).*(x(d:end)-mu(t)),1,'omitnan');
                end
                tmpss(isnan(tmpss)) = 0;
                ss(t,:) = tmpss;
            end
            suffstat.mu{f} = ones(D,1)*mean(mu,1);
            suffstat.ss{f} = toeplitz( sum(ss .* repmat(n,1,D),1)/sum(n) );
            suffstat.n{f} = D;
            
            eig_ss = eig(suffstat.ss{f});
            if any(eig_ss<0)
                lambdas = 0.1.^(10:-1:1);
                min_eig_ss = min(eig_ss);
                regularizer = lambdas(find(lambdas > abs(min_eig_ss), 1, 'first'));
                
                suffstat.ss{f} = suffstat.ss{f} + eye(D)*regularizer;
                
                warning('Regularization added to feature %u to ensure positive-definite covariance: %f', f, regularizer);
                
                if any(eig(suffstat.ss{f}) < 0)
                    error('regularization didn''t work!')
                end
            end     
        end

    case 'lognormal'
        % Take log of input
        xs = cellfun(@log, xs, 'UniformOutput', false);
        
        suffstat = [];
        suffstat.mu = cell(nfeature,1); % mu: mean
        suffstat.ss = cell(nfeature,1); % ss: sum of squared deviation from mean, "scatter matrix"
        suffstat.n = cell(nfeature,1);  % n: count
        
        for f = 1:nfeature
            mu = zeros(ninput,1);
            ss = zeros(ninput,D);
            n = zeros(ninput,1);
            for t = 1:ninput
                mu(t) = mean(xs{t}(:,f),1,'omitnan');
                tmpss = zeros(1,D);
                n(t) = size(xs{t},1);
                for d = 1:D
                    x = xs{t}(:,f);
                    tmpss(d) = mean((x(1:end-d+1)-mu(t)).*(x(d:end)-mu(t)),1,'omitnan');
                end
                tmpss(isnan(tmpss)) = 0;
                ss(t,:) = tmpss;
            end
            suffstat.mu{f} = ones(D,1)*mean(mu,1);
            suffstat.ss{f} = toeplitz( sum(ss .* repmat(n,1,D),1)/sum(n) );
            suffstat.n{f} = D;
        end
        
    case 'gmm'
        if D~=1
            warning('Setting D=1 for gmm distribution')
            D = 1;
        end
        suffstat = [];
        suffstat.mu = cell(nfeature,1); % mu: mean
        suffstat.sigma = cell(nfeature,1); % ss: sum of squared deviation from mean, "scatter matrix"
        suffstat.n = cell(nfeature,1);  % n: count

        for f = 1:nfeature
            mu = zeros(ninput,1);
            ss = zeros(ninput,D);
            n = zeros(ninput,1);
            for t = 1:ninput
                mu(t) = mean(xs{t}(:,f),1,'omitnan');
                tmpss = zeros(1,D);
                n(t) = size(xs{t},1);
                for d = 1:D
                    x = xs{t}(:,f);
                    tmpss(d) = mean((x(1:end-d+1)-mu(t)).*(x(d:end)-mu(t)),1,'omitnan');
                end
                tmpss(isnan(tmpss)) = 0;
                ss(t,:) = tmpss;
            end
            
            % prior has single component with mean and variance of input
            suffstat.n{f} = nan(1,max_ncomp);
            suffstat.n{f}(1) = D;
            
            suffstat.mu{f} = nan(1,max_ncomp);
            suffstat.mu{f}(1) = ones(D,1)*mean(mu,1);
            
            suffstat.sigma{f} = nan(1,max_ncomp);
            suffstat.sigma{f}(1) = toeplitz( sum(ss .* repmat(n,1,D),1)/sum(n) );
            
            suffstat.pi{f} = zeros(1,max_ncomp);
            suffstat.pi{f}(1) = 1;
            
            suffstat.sp{f} = zeros(1,max_ncomp);
            suffstat.sp{f}(1) = 1;
            
            suffstat.k{f} = 1;
        end
        
    case 'poisson'
        suffstat = [];
        suffstat.lambda = cell(nfeature);
        suffstat.n = cell(nfeature,1);
        
        for f = 1:nfeature
            lambda = zeros(ninput,1);
            for i = 1:ninput
                n = 0;
                for t = 1:length(xs{i}(:,f)) - D
                    spikesum = sum(xs{i}((1:D)+t-1,f));
                    lambda(i) = (lambda(i)*n + spikesum)/(n+1);
                    n = n+1;
                end
            end
            suffstat.lambda{f} = mean(lambda);
            suffstat.n{f} = 1;
        end
        
    otherwise
        error(['Unsupported distribution: ' dist]);
end

end