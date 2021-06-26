function [out] = run_DREX_model(x, params)
% Usage: [out] = run_DREX_model(x, params)
%
% D-REX model for Dynamic statistical REgularity eXtraction

% Assumes observations come from an underlying probabilitity distribution
% (specified in params) with unknown parameters, builds robust predictions
% by collecting sufficient statistics and calculating beliefs across
% multiple context windows causally. Distributions currently supported:
% Gaussian, Log-Normal, Gaussian Mixture Model (GMM), Poisson. Gaussian and
% Log-Normal have options temporal dependence between inputs, GMM and
% Poisson assume independent inputs.
%
% NOTE: If input has multiple features (i.e., size(x,2)>1), predictions
% along each feature are multiplied before updating beliefs.
%
% ===INPUT===
%     x           input sequence of observations (dim: time x feature)
%     params      structure with model parameters (see below for more info)
%
% ===OUTPUT===
%     out         output structure with sequential model results (see below for more info)
%
%
% * Params structure
%     distribution    Distribution choice: 'gaussian','lognormal','gmm', or 'poisoon' (default='gaussian')
%     D               temporal dependence (or interval size for Poisson), integer (default=1, 50 for Poisson), 
%     prior           structure with priors for sufficient statistics (see below)
%     hazard          prior probability of change, scalar (constant) or vector (time-varying) (default=0.01)
%     obsnz           observation noise for each feature, vector (default=0.0)
%     memory          maximum number of context hypotheses, integer (default=inf)
%     maxhyp          maximum number of simultaneous context hypotheses, integer (default=inf)
%
% * Priors structure, depends on distribution choice, for example for 'gaussian':
% Each field is a cell array with a cell for each feature
%     mu{f}       prior mean (size: D x 1)
%     ss{f}       prior sum of squares (size: D x D)
%     n{f}        prior observation count (size: 1 x 1)
% Note: same structure as output of function 'estimate_suffstat.m'
%
% * Output structure
%     surprisal           surprisal due to each observation in bits (dim: time x feature)
%     joint_surprisal     surprisal across features (dim: time x 1), features combined with logical-AND (i.e., product of predictive probability across features)
%     context_beliefs     posterior beliefs for context hypotheses (dim: context-boundary x time)
%     prediction_params    parameters of predictive distribution at each time (dim: time x feature)
%
% v3
% Benjamin Skerritt-Davis
% bsd@jhu.edu

[ntime, nfeature] = size(x);

if isfield(params,'changeprior')
    error('changeprior -> hazard in params')
end

% Parameters
if ~isfield(params,'distribution'),  distribution = 'gaussian';     else, distribution = params.distribution; end
if ~isfield(params,'prior'),         error('set prior');            else, prior = params.prior; end
if ~isfield(params,'hazard'),        hazard = 0.01;                 else, hazard = params.hazard; end
if ~isfield(params,'D'),             D = 1;                         else, D = params.D; end
if ~isfield(params,'obsnz'),         obsnz = zeros(nfeature,1);     else, obsnz = params.obsnz; end
if ~isfield(params,'memory'),        memory = inf;                  else, memory = params.memory; end
if ~isfield(params,'maxhyp'),        maxhyp = inf;                  else, maxhyp = params.maxhyp; end
if ~isfield(params,'predscale'),     predscale = 1e-3;              else, predscale = params.predscale; end

% check input and parameters match
if size(x,2) > size(x,1); error('input should be time x feature'); end
if size(x,1)==0 || numel(x)==0; error('input has zero length'); end
if nfeature ~= length(obsnz); error('obsnz and nfeature mismatch'); end
if ~strcmp(distribution, 'poisson') && any([prior.n{:}] < D); error('prior n''s must all be >= D'); end
if isinf(memory) || memory > ntime+1; memory = ntime+1; end
if memory < 2; error('memory must be greater than 1'); end

% Distribution-specific parameters and parameter checks
switch distribution
    case 'gmm'
        % max number of components
        if ~isfield(params,'max_ncomp'), max_ncomp = 10; else, max_ncomp = params.max_ncomp; end
        % Thresh for creating new comp. Lower threshold means new inputs
        % are more likely to be incorporated into existing components.
        if ~isfield(params,'beta'), beta = 0.001; else, beta = params.beta; end
        if D ~= 1
            error('Temporal dependence not supported. Set D=1 for GMM distribution.');
        end
    case 'poisson' 
        % For Poisson distribution, D is the temporal interval into the past for counting events
        if ~isfield(params,'D'), D = 50; else, D = params.D; end
end

% If hazard rate is scalar (constant), vectorize
if numel(hazard)==1
    hazard = hazard*ones(size(x,1),1);
end


%=== INITIALIZE ==========================================

% Initialize conditioning observations for D>1
cond_obs = nan(D-1,nfeature);       

% Initialize output arrays
surprisal = zeros(ntime,nfeature);  % Surprisal at each time for each feature
joint_surprisal = zeros(ntime,1);   % Surprisal at each time across features
B = zeros(memory, ntime+1);         % Beliefs, or context posterior, at each time (dim: context_hypothesis x time)
B(1,1) = 1;                         % context_length=0 at time=0 (i.e., sequence begins at first observation)
prediction_theta = cell(ntime,1);


% Initialize sufficient statistics with priors
suffstat = [];
for f = 1:nfeature
    switch distribution
        case 'gaussian'
            try
                % Initialize with NaNs
                suffstat.n{f} = nan(memory,1);      % obs count
                suffstat.mu{f} = nan(D,memory);     % mean
                suffstat.ss{f} = nan(D,D,memory);   % sum of squared deviations
                
                % Initialize first hypothesis with prior
                suffstat.n{f}(1) = prior.n{f};
                suffstat.mu{f}(:,1) = prior.mu{f};
                suffstat.ss{f}(:,:,1) = prior.ss{f};
            catch err
                getReport(err)
                error('Issue with prior and Gaussian sufficient statistics');
            end
        case 'lognormal'
            try
                % Initialize with NaNs
                suffstat.n{f} = nan(memory,1);      % obs count
                suffstat.mu{f} = nan(D,memory);     % mean
                suffstat.ss{f} = nan(D,D,memory);   % sum of squared deviations
                
                % Initialize first hypothesis with prior
                suffstat.n{f}(1) = prior.n{f};
                suffstat.mu{f}(:,1) = prior.mu{f};
                suffstat.ss{f}(:,:,1) = prior.ss{f};
            catch err
                getReport(err)
                error('Issue with prior and Log-normal sufficient statistics');
            end
        case 'gmm'
            try
                % Initialize with NaNs
                suffstat.k{f} = nan(memory, 1);          % num of components
                suffstat.n{f} = nan(memory, max_ncomp);           % obs count
                suffstat.mu{f} = nan(memory, max_ncomp);   % mean
                suffstat.sigma{f} = nan(memory, max_ncomp);   % sum of squared deviations
                suffstat.pi{f} = zeros(memory, max_ncomp);   % component weight
                suffstat.sp{f} = nan(memory, max_ncomp);   % component likelihood
                
                % Initialize first hyp with prior
                suffstat.k{f}(1) = prior.k{f};
                suffstat.n{f}(1,:) = prior.n{f};
                suffstat.mu{f}(1,:) = prior.mu{f};
                suffstat.sigma{f}(1,:) = prior.sigma{f};
                suffstat.pi{f}(1,:) = prior.pi{f};
                suffstat.sp{f}(1,:) = prior.sp{f};
            catch err
                getReport(err)
                keyboard;
                error('Issues with prior and GMM sufficient statistics');
            end
        case 'poisson'
            try
                % Initialize with NaNs
                suffstat.n{f} = nan(memory,1);      % obs count
                suffstat.lambda{f} = nan(memory,1);     % mean
                
                % Initialize first hypothesis with prior
                suffstat.n{f}(1) = prior.n{f};
                suffstat.lambda{f}(1) = prior.lambda{f};
            catch err
                getReport(err)
                error('Issues with prior and Poisson sufficient statistics');
            end
        otherwise
            error(['Unsupported distribution: ' distribution]);
    end
end


% =================
%     MAIN LOOP 
% =================
for t = 1:ntime
    
    % ==== OBSERVE: new input ======================================================
    obs = x(t,:); 
    
    
    % ==== PREDICT: compute context-specific predictive probs of new input =========
    switch distribution
        case 'gaussian'
            pred = predict_GAUSSIAN(obs, cond_obs, suffstat, B(:,t), D, obsnz, predscale);
        case 'lognormal'
            pred = predict_LOGNORMAL(obs, cond_obs, suffstat, B(:,t), D, obsnz, predscale);
        case 'gmm'
            pred = predict_GMM(obs, suffstat, B(:,t), obsnz, predscale);
        case 'poisson'
            pred = predict_POISSON(obs, cond_obs, suffstat, B(:,t), predscale);
        otherwise
            error(['Unsupported distribution: ' distribution]);
    end
    
    % Extra prediction info: expected value, error, predictive distribution
    % params (for computing full predictive distribution, \Psi)
    if isempty(pred)
        prediction_theta{t} = prediction_theta{t-1};
        pflds = fields(prediction_theta{t});
        for f = 1:length(pflds)
            prediction_theta{t}.(pflds{f})(end+1,:) = prediction_theta{t}.(pflds{f})(end,:);
        end
    else
        prediction_theta{t} = pred.ss;
    end
    
    % Calculate Surprisal 
    if isnan(obs) % no input, no surprisal
        surprisal(t,:) = nan;
        joint_surprisal(t,:) = nan;
    else
        surprisal(t,:) = -1*log2(pred.prob'*B(1:min(t,memory),t));
        joint_surprisal(t,:) = -1*log2(prod(pred.prob,2)'*B(1:min(t,memory),t));
    end
    
    % ==== UPDATE context-beliefs with predictive probabilities ===========
    % Combine prediction across features (i.e., probabilistic-AND across
    % features) to update context beliefs
    pp = [];
    if ~isempty(pred)
        pp = prod(pred.prob,2);
    end
    B = update_context_posterior(B, pp, hazard(t), t, maxhyp);
    
    % ==== UPDATE sufficient statistics with new observation ==============
    switch distribution
        case 'gaussian'
            [cond_obs, suffstat] = update_GAUSSIAN(obs, cond_obs, D, suffstat, B(:,t), prior, obsnz);
        case 'lognormal'
            [cond_obs, suffstat] = update_LOGNORMAL(obs, cond_obs, D, suffstat, B(:,t), prior, obsnz);
        case 'gmm'
            suffstat = update_GMM(obs, suffstat, pred, prior, obsnz, beta*predscale);
        case 'poisson'
            [cond_obs, suffstat] = update_POISSON(obs, cond_obs, suffstat, B(:,t), prior);
        otherwise
            error(['Unsupported distribution: ' distribution]);
    end
    
end


% ========= OUTPUT ==========
out.distribution = distribution;
out.surprisal = surprisal;
out.joint_surprisal = joint_surprisal;
out.context_beliefs = B;
out.prediction_params = prediction_theta;

end


%%   *****************************************
%    |            SUB-FUNCTIONS              |
%    *****************************************

function R = update_context_posterior(R, pp, hazard, t, maxhyp)
% Update beliefs with predictive probabilities
% pp: predictive probabilities
% hazard: hazard rate
% t: current time

memory = size(R,1);

% floor beliefs less than epsilon to prevent precision errors
R(R(:,t)<1e-300, t) = 0;

% If no prediction, change prob is 0.
if isempty(pp)
    R(1:min(t,memory-1), t+1) = R(1:min(t,memory-1), t); % Change prob
    R(min(t+1,memory),t+1) = 0; % Growth prob
    return;
end

try
    if memory <= t
        % Growth prob: P(c_t=1:t, x_t:t)
        R(1:(memory-1),t+1) = pp(2:end) .* (1-hazard) .* R(2:memory,t);
        R(1,t+1) = R(1,t+1) + pp(1) .* (1-hazard) .* R(1,t);
        % Change prob: P(c_t=0, x_1:t)
        R(memory,t+1) = sum(pp(1:end) .* hazard .* R(1:memory,t));
    else
        % Growth prob: P(c_t=1:t, x_t:t)
        R(1:t,t+1) = pp .* (1-hazard) .* R(1:t,t);
        % Change prob: P(c_t=0, x_1:t)
        R(t+1,t+1) = sum(pp .* hazard .* R(1:t,t));
    end
catch
    keyboard;
end
% Check context posterior
if any(R(:) < 0)
    disp('ERROR with context posterior');
    keyboard;
end

% prune lowest prob context hypothesis if exdeeded maxhyp
hypidx = find(R(1:min(t,memory-1),t+1) > 0);
if maxhyp < inf && length(hypidx) >= maxhyp
    [~,worsthypidx] = min(R(hypidx,t+1));
    R(hypidx(worsthypidx),t+1) = 0;
end

% Normalize posterior to sum to 1
R(:,t+1) = R(:,t+1) / sum(R(:,t+1));

end

% =====================================================================
%                    DISTRIBUTION: GAUSSIAN
% =====================================================================

% ==== PREDICT for each context hypothesis ============================
function p = predict_GAUSSIAN(obs, cond_obs, suffstat, beliefs, D, obsnz, scale)
% pred: vector of predictive probabilities
% condSS: conditional sufficient statistics

% Skip prediction for any hyps with belief=0
keephyp = find(beliefs > 1e-300);

% If silent/missing observation, no prediction to make
if any(isnan(obs) | isempty(obs))
    % NOTE: assumes observation silent/missing simultaneously for all
    % features
    p = [];
    return;
end

nhyp = sum(~isnan(suffstat.n{1})); % number of hypotheses incl. ones with belief=0
nkeephyp = length(keephyp); % number of hypotheses with belief>0
nfeature = length(suffstat.n);
pred = zeros(nkeephyp,nfeature);  % predictive probabilities of new observation

% sufficient statistics
muT = suffstat.mu;
ssT = suffstat.ss;
nT = suffstat.n;

% Loop over features, calc cond distribution and predictions for each context hypotheses
nCond = zeros(nkeephyp,nfeature);         % conditional count
muCond = zeros(nkeephyp,nfeature);         % conditional mean
covCond = zeros(nkeephyp,nfeature);      % conditional (co)variance

for f = 1:nfeature
    % condition current observation on past d-1 observations, 
    % Note: ensure conditioning observations are not all NaNs (assumes same
    % for all input features)
    if D>1 && sum(isnan(cond_obs(:,1))) < length(cond_obs(:,1))
        for hh = 1:nkeephyp
            h = keephyp(hh);
            sigmaJoint = ssT{f}(:,:,h)*(nT{f}(h)+1)/(nT{f}(h)*(nT{f}(h)-D+1));
            muJoint = muT{f}(:,h);
            nuJoint = nT{f}(h)-D+1;
            
            devFromMean = cond_obs(:,f) - muJoint(1:D-1);
            % Replace NaNs with 0 to marginalize over missing context
            devFromMean(isnan(devFromMean)) = 0;
            
            nCond(hh,f) = nuJoint+D-1;
            z = sigmaJoint(D,1:D-1)/sigmaJoint(1:D-1,1:D-1);
            muCond(hh,f) = muJoint(D) + z*devFromMean;
            covCond(hh,f) = ((nuJoint + devFromMean'/sigmaJoint(1:D-1,1:D-1)*devFromMean)/nCond(hh,f))*...
                (sigmaJoint(D,D) - z*sigmaJoint(1:D-1,D));
            
            if any(~isreal(covCond) | ~isreal(muCond))
                warning('ERROR with predictive probabilities')
                keyboard;
            end
        end
        
    else % D=1, no conditioning
        for hh = 1:nkeephyp
            h = keephyp(hh);
            covCond(hh,f) = ssT{f}(1,1,h)*(nT{f}(h)+1)/(nT{f}(h)*(nT{f}(h)));
            muCond(hh,f) = muT{f}(h);
            nCond(hh,f) = nT{f}(h);
        end
    end
    % Calculate predictive probability of new observation given each hypothesis
    pred(:,f) = studentpdf(obs(f), muCond(:,f), covCond(:,f) + obsnz(f)^2, nCond(:,f))*scale;
end

% Put predictions back into array with prediction=0 for belief=0 hypotheses
condSS.mu = zeros(nhyp,nfeature);
condSS.mu(keephyp,:) = muCond;
condSS.cov = zeros(nhyp,nfeature);
condSS.cov(keephyp,:) = covCond;
condSS.n = zeros(nhyp,nfeature);
condSS.n(keephyp,:) = nCond;
tmp = pred;
pred = zeros(nhyp,nfeature);
pred(keephyp,:) = tmp;

% Prob ceiling at 1 (in case of variance << 1)
if any(pred > 1)
    error('Predictive prob greater than one. Decrease predscale to combat this.');
end

% Check predictive probabilities
if any(isnan(pred) | ~isreal(pred))
    warning('ERROR with predictive probabilities')
    keyboard;
end

p = [];
p.prob = pred;
% beliefs = beliefs(1:length(condSS.mu))';
% p.expected = beliefs * condSS.mu;
% p.error = abs(p.expected - obs);
p.ss = condSS;

end

% ==== UPDATE sufficient statistics with new observation ==============
function [cond_obs, suffstat] = update_GAUSSIAN(obs, cond_obs, D, suffstat, beliefs, prior, obsnz)
% If prior==[], only update statistics.

nfeature = length(suffstat.n);
nhyp = sum(~isnan(suffstat.n{1}));
memory = length(suffstat.n{1});

% Skip update for any hyps with belief=0
keephyp = find(beliefs > 1e-300);
nkeephyp = length(keephyp);

% Replace NaNs with 0s to marginalize over missing context
obs_w_context = [cond_obs; obs];
obs_w_context(isnan(obs_w_context)) = 0;

for f = 1:nfeature
    
    % Update statistics, unless input obs is empty/missing
    if ~any(isnan(obs) | isempty(obs))
        n_update = suffstat.n{f}(keephyp) + 1;
        mu_update = (repmat(suffstat.n{f}(keephyp),1,D)'.*suffstat.mu{f}(:,keephyp) + repmat(obs_w_context(:,f),1,nkeephyp))./repmat(n_update,1,D)';
        
        tmpcov = zeros(D,D,nkeephyp);
        for hh = 1:nkeephyp
            h = keephyp(hh);
            tmpcov(:,:,hh) = ((obs_w_context(:,f)-suffstat.mu{f}(:,h))*(obs_w_context(:,f)-suffstat.mu{f}(:,h))' + eye(D)*obsnz(f)^2);
        end
        
        suffstat.ss{f}(:,:,keephyp) = suffstat.ss{f}(:,:,keephyp) + tmpcov.*repmat(shiftdim(suffstat.n{f}(keephyp)./n_update,-2),D,D,1);
        suffstat.mu{f}(:,keephyp) = mu_update;
        suffstat.n{f}(keephyp) = n_update;
        
        % clear suffstats for hyps with beliefs=0
        suffstat.ss{f}(:,:,~ismember(1:nhyp,keephyp)) = 0;
        suffstat.mu{f}(:,~ismember(1:nhyp,keephyp)) = 0;
        suffstat.n{f}(~ismember(1:nhyp,keephyp)) = 0;
    end
    
    for h = 1:size(suffstat.ss{f},3)
        if suffstat.n{f}(h)==0 || isnan(suffstat.n{f}(h))
            continue
        else
            try
                if any(eig(suffstat.ss{f}(:,:,h))<=0)
                    keyboard
                end
            catch
                keyboard
            end
        end
    end
    
    % Concatenating new hypothesis
    if ~isempty(prior)
        if nhyp < memory
            % add prior as newest hypothesis
            suffstat.n{f}(nhyp+1) = prior.n{f};
            suffstat.mu{f}(:,nhyp+1) = prior.mu{f};
            suffstat.ss{f}(:,:,nhyp+1) = prior.ss{f};
        else
            % remove oldest hypothesis and add prior as newest hypothesis
            suffstat.n{f} = cat(1,suffstat.n{f}(2:end),prior.n{f});
            suffstat.mu{f} = cat(2,suffstat.mu{f}(:,2:end), prior.mu{f});
            suffstat.ss{f} = cat(3,suffstat.ss{f}(:,:,2:end), prior.ss{f});
        end
    end
end

% increment conditioning observations to include new observation
cond_obs = [cond_obs; obs];
cond_obs(1,:) = [];

end


% =====================================================================
%                    DISTRIBUTION: LOG-NORMAL
% =====================================================================

% ==== PREDICT for each context hypothesis ============================
function p = predict_LOGNORMAL(obs, cond_obs, suffstat, beliefs, D, obsnz, scale)
% pred: vector of predictive probabilities
% condSS: conditional sufficient statistics

% Take log of new observation and context
obs = log(obs);
cond_obs = log(cond_obs);

% Skip prediction for any hyps with belief=0
keephyp = find(beliefs > 1e-300);

% If silent/missing observation, no prediction to make
if any(isnan(obs) | isempty(obs))
    % NOTE: assumes observation silent/missing simultaneously for all
    % features
    p = [];
    return;
end

nhyp = sum(~isnan(suffstat.n{1})); % number of hypotheses incl. ones with belief=0
nkeephyp = length(keephyp); % number of hypotheses with belief>0
nfeature = length(suffstat.n);
predprobs = zeros(nkeephyp,nfeature);  % predictive probabilities of new observation

% sufficient statistics
muT = suffstat.mu;
ssT = suffstat.ss;
nT = suffstat.n;

% Loop over features, calc cond distribution and predictions for each context hypotheses
nCond = zeros(nkeephyp,nfeature);         % conditional count
muCond = zeros(nkeephyp,nfeature);         % conditional mean
covCond = zeros(nkeephyp,nfeature);      % conditional (co)variance

for f = 1:nfeature
    % condition current observation on past d-1 observations
    if D>1 && sum(isnan(cond_obs(:,1))) < length(cond_obs(:,1))
        for hh = 1:nkeephyp
            h = keephyp(hh);
            sigmaJoint = ssT{f}(:,:,h)*(nT{f}(h)+1)/(nT{f}(h)*(nT{f}(h)-D+1));
            muJoint = muT{f}(:,h);
            nuJoint = nT{f}(h)-D+1;
            
            devFromMean = cond_obs(:,f) - muJoint(1:D-1);
            % Replace NaNs with 0 to marginalize over missing context
            devFromMean(isnan(devFromMean)) = 0;
            
            nCond(hh,f) = nuJoint+D-1;
            z = sigmaJoint(D,1:D-1)/sigmaJoint(1:D-1,1:D-1);
            muCond(hh,f) = muJoint(D) + z*devFromMean;
            covCond(hh,f) = ((nuJoint + devFromMean'/sigmaJoint(1:D-1,1:D-1)*devFromMean)/nCond(hh,f))*...
                (sigmaJoint(D,D) - z*sigmaJoint(1:D-1,D));
        end
        
    else % D=1, no conditioning
        for hh = 1:nkeephyp
            h = keephyp(hh);
            covCond(hh,f) = ssT{f}(1,1,h)*(nT{f}(h)+1)/(nT{f}(h)*(nT{f}(h)));
            muCond(hh,f) = muT{f}(h);
            nCond(hh,f) = nT{f}(h);
        end
    end
    % Calculate predictive probability of new observation given each hypothesis
    predprobs(:,f) = studentpdf(obs(f), muCond(:,f), covCond(:,f) + obsnz(f)^2, nCond(:,f)) * scale;
end

% Put predictions back into array with prediction=0 for belief=0 hypotheses
condSS.mu = zeros(nhyp,nfeature);
condSS.mu(keephyp,:) = muCond;
condSS.cov = zeros(nhyp,nfeature);
condSS.cov(keephyp,:) = covCond;
condSS.n = zeros(nhyp,nfeature);
condSS.n(keephyp,:) = nCond;
tmp = predprobs;
predprobs = zeros(nhyp,nfeature);
predprobs(keephyp,:) = tmp;


% Prob ceiling at 1 (in case of variance << 1)
if any(predprobs > 1)
    error('Predictive prob greater than one. Decrease predscale to combat this.');
end

% Check predictive probabilities
if any(isnan(predprobs) | ~isreal(predprobs))
    warning('ERROR with predictive probabilities')
    keyboard;
end

p = [];
p.prob = predprobs;
% p.expected = beliefs(1:length(condSS.mu))' * exp(condSS.mu+0.5*condSS.cov);
% p.error = abs(exp(p.expected) - exp(obs));
p.ss = condSS;
end

% ==== UPDATE sufficient statistics with new observation ==============
function [cond_obs, suffstat] = update_LOGNORMAL(obs, cond_obs, D, suffstat, beliefs, prior, obsnz)
% If prior==[], only update statistics.

% Take log of new observation and context
origobs = obs;
origcontext = cond_obs;
obs = log(obs);
cond_obs = log(cond_obs);


nfeature = length(suffstat.n);
nhyp = sum(~isnan(suffstat.n{1}));
memory = length(suffstat.n{1});

% Skip update for any hyps with belief=0
keephyp = find(beliefs > 1e-300);
nkeephyp = length(keephyp);

% Replace NaNs with 0s to marginalize over missing context
obs_w_context = [cond_obs; obs];
obs_w_context(isnan(obs_w_context)) = 0;

for f = 1:nfeature
    
    % Update statistics, unless input obs is empty/missing
    if ~any(isnan(obs) | isempty(obs))
        n_update = suffstat.n{f}(keephyp) + 1;
        mu_update = (repmat(suffstat.n{f}(keephyp),1,D)'.*suffstat.mu{f}(:,keephyp) + repmat(obs_w_context(:,f),1,nkeephyp))./repmat(n_update,1,D)';
        
        tmpcov = zeros(D,D,nkeephyp);
        for hh = 1:nkeephyp
            h = keephyp(hh);
            tmpcov(:,:,hh) = ((obs_w_context(:,f)-suffstat.mu{f}(:,h))*(obs_w_context(:,f)-suffstat.mu{f}(:,h))' + eye(D)*obsnz(f)^2);
        end
        
        suffstat.ss{f}(:,:,keephyp) = suffstat.ss{f}(:,:,keephyp) + tmpcov.*repmat(shiftdim(suffstat.n{f}(keephyp)./n_update,-2),D,D,1);
        suffstat.mu{f}(:,keephyp) = mu_update;
        suffstat.n{f}(keephyp) = n_update;
        
        % clear suffstats for hyps with beliefs=0
        suffstat.ss{f}(:,:,~ismember(1:nhyp,keephyp)) = 0;
        suffstat.mu{f}(:,~ismember(1:nhyp,keephyp)) = 0;
        suffstat.n{f}(~ismember(1:nhyp,keephyp)) = 0;
    end
    
    % Concatenating new hypothesis
    if ~isempty(prior)
        if nhyp < memory
            % add prior as newest hypothesis
            suffstat.n{f}(nhyp+1) = prior.n{f};
            suffstat.mu{f}(:,nhyp+1) = prior.mu{f};
            suffstat.ss{f}(:,:,nhyp+1) = prior.ss{f};
        else
            % remove oldest hypothesis and add prior as newest hypothesis
            suffstat.n{f} = cat(1,suffstat.n{f}(2:end),prior.n{f});
            suffstat.mu{f} = cat(2,suffstat.mu{f}(:,2:end), prior.mu{f});
            suffstat.ss{f} = cat(3,suffstat.ss{f}(:,:,2:end), prior.ss{f});
        end
    end
end

% increment context to include new observation
cond_obs = [origcontext; origobs];
cond_obs(1,:) = [];

end

% =====================================================================
%              DISTRIBUTION: GAUSSIAN MIXTURE MODEL (GMM)
% =====================================================================

% ==== PREDICT for each context hypothesis ============================
function p = predict_GMM(obs, suffstat, beliefs, obsnz, scale)
% pred: vector of predictive probabilities
% condSS: conditional sufficient statistics

% Skip prediction for any hyps with belief=0
keephyp = find(beliefs > 1e-300);

% If silent/missing observation, no prediction to make
if any(isnan(obs) | isempty(obs))
    % NOTE: assumes observation silent/missing simultaneously for all
    % features
    p = [];
    return;
end

nhyp = sum(~isnan(suffstat.k{1})); % number of hypotheses incl. ones with belief=0
nkeephyp = length(keephyp); % number of hypotheses with belief>0
nfeature = length(suffstat.n);
component_probs = cell(nfeature,1);
predprobs = zeros(nkeephyp,nfeature);  % predictive probabilities of new observation

% sufficient statistics
muT = suffstat.mu;
sigmaT = suffstat.sigma;
spT = suffstat.sp;
piT = suffstat.pi;

for f = 1:nfeature
    component_probs{f} = studentpdf(obs(f), muT{f}(keephyp,:), sigmaT{f}(keephyp,:)+obsnz(f)^2, spT{f}(keephyp,:)) * scale; % dim: hypothesis x component
    predprobs(:,f) = sum(component_probs{f} .* piT{f}(keephyp,:),2,'omitnan');
end

% Put predictions back into array with prediction=0 for belief=0 hypotheses
tmp = predprobs;
predprobs = zeros(nhyp,nfeature);
predprobs(keephyp,:) = tmp;

tmp = component_probs;
for f = 1:nfeature
    component_probs{f} = zeros(nhyp,size(tmp{f},2));
    component_probs{f}(keephyp,:) = tmp{f};
end

% Prob ceiling at 1 (in case of variance << 1)
if any(predprobs > 1)
    error('A predictive prob is greater than one. Decrease predscale to combat this.');
end

% Check predictive probabilities
if any(isnan(predprobs) | ~isreal(predprobs))
    warning('ERROR with predictive probabilities')
    keyboard;
end

p = [];
p.prob = predprobs;
p.component_probs = component_probs;
% p.expected = 0; %beliefs(1:length(condSS.mu))' * condSS.mu;
% p.error = 0; %abs(p.expected - obs);
p.ss = [];
flds = fields(suffstat);
for f = 1:nfeature
    for fld = 1:length(flds)
        p.ss.(flds{fld}){f} = suffstat.(flds{fld}){f}(1:nhyp,:);
    end
end

end

% ==== UPDATE sufficient statistics with new observation ==============
function suffstat = update_GMM(obs, suffstat, pred, prior, obsnz, beta)
% If prior==[], only update statistics.


nfeature = length(suffstat.n);
memory = length(suffstat.n{1});

max_comp = size(suffstat.mu{1},2);

% TODO: Replace NaNs with 0s to marginalize over missing context

for f = 1:nfeature
    
    % Update statistics, unless input obs is empty/missing
    if ~any(isnan(obs) | isempty(obs))
        
        % Create new component
        nhyp = size(pred.prob,1);
        try
            create_comp = (max(pred.component_probs{f},[],2,'omitnan') < beta) & (suffstat.k{f}(1:nhyp) < max_comp);
        catch
            keyboard;
        end
        % Update existing component components
        % Calculate component likelihood given current observation
        lik = suffstat.pi{f}(1:nhyp,:) .* pred.component_probs{f};
        lik = lik ./ repmat(sum(lik,2,'omitnan'),1,size(lik,2));
        for h = 1:nhyp
            kh = suffstat.k{f}(h); % num of comps for current hypothesis
            if create_comp(h)
                % obs comes from new component with prob 1
                lik(h,:) = 0;
                lik(h,kh+1) = 1;
                suffstat.sp{f}(h,kh+1) = 0;
                suffstat.n{f}(h,kh+1) = 0;
                suffstat.mu{f}(h,kh+1) = obs(f);
                suffstat.sigma{f}(h,kh+1) = prior.sigma{f}(1);
            end
        end
        
        % Update likelihood accumulatos and priors
        sp_update = suffstat.sp{f}(1:nhyp,:) + lik;
        w = lik ./ sp_update; % updated weights for each component
        
        % Update component means
        mu_update = suffstat.mu{f}(1:nhyp,:) + w.*(obs(f) - suffstat.mu{f}(1:nhyp,:));
        
        % Update component variance
        sigma_update = suffstat.sigma{f}(1:nhyp,:) + w.*((obs(f) - suffstat.mu{f}(1:nhyp,:)).*(obs(f)-mu_update) + obsnz(f)^2 - suffstat.sigma{f}(1:nhyp,:));
        
        % Update component obs count
        n_update = suffstat.n{f}(1:nhyp,:) + 1;
        
        % Reset suff stats for new components
        k_update = suffstat.k{f}(1:nhyp)+create_comp;
        mu_update(create_comp, k_update(create_comp)) = obs(f);
        sigma_update(create_comp, k_update(create_comp)) = prior.sigma{f}(1);
        
        % Update component priors
        pi_update = sp_update ./ repmat(sum(sp_update,2,'omitnan'),1,size(sp_update,2));
        
        
        suffstat.k{f}(1:nhyp) = k_update;
        suffstat.n{f}(1:nhyp,:) = n_update;
        suffstat.mu{f}(1:nhyp,:) = mu_update;
        suffstat.sigma{f}(1:nhyp,:) = sigma_update;
        suffstat.pi{f}(1:nhyp,:) = pi_update;
        suffstat.sp{f}(1:nhyp,:) = sp_update;
        
        
        % Concatenating new hypothesis
        if ~isempty(prior)
            
            if nhyp == memory
                % remove oldest hypothesis
                suffstat.k{f} = suffstat.k{f}(2:end);
                suffstat.n{f} = suffstat.n{f}(2:end,:);
                suffstat.mu{f} = suffstat.mu{f}(2:end,:);
                suffstat.sigma{f} = suffstat.sigma{f}(2:end,:);
                suffstat.pi{f} = suffstat.pi{f}(2:end,:);
                suffstat.sp{f} = suffstat.sp{f}(2:end,:);
                
                nhyp = memory - 1;
            end
            
            
            % add prior as newest hypothesis
            suffstat.k{f}(nhyp+1) = prior.k{f};
            suffstat.n{f}(nhyp+1,:) = prior.n{f};
            suffstat.mu{f}(nhyp+1,:) = prior.mu{f};
            suffstat.sigma{f}(nhyp+1,:) = prior.sigma{f};
            suffstat.pi{f}(nhyp+1,:) = prior.pi{f};
            suffstat.sp{f}(nhyp+1,:) = prior.sp{f};
            
        end
    end
end

end


% =====================================================================
%              DISTRIBUTION: POISSON
% =====================================================================

% ==== PREDICT for each context hypothesis ============================
function p = predict_POISSON(obs, cond_obs, suffstat, beliefs, scale)
% pred: vector of predictive probabilities
% condSS: conditional sufficient statistics

% Skip prediction for any hyps with belief=0
keephyp = find(beliefs > 1e-300);

% If silent/missing observation, no prediction to make
if any(isnan(obs) | isempty(obs))
    % NOTE: assumes observation silent/missing simultaneously for all
    % features
    p = [];
    return;
end

input = sum([cond_obs; obs],'omitnan');

nhyp = sum(~isnan(suffstat.n{1})); % number of hypotheses incl. ones with belief=0
nkeephyp = length(keephyp); % number of hypotheses with belief>0
nfeature = length(suffstat.n);
pred = zeros(nkeephyp,nfeature);  % predictive probabilities of new observation

% sufficient statistics
lambdaT = suffstat.lambda;
nT = suffstat.n;

% Loop over features, calc cond distribution and predictions for each context hypotheses
nCond = zeros(nkeephyp,nfeature);         % conditional count
lambdaCond = zeros(nkeephyp,nfeature);    % conditional mean


% Calculate predictive probability of new observation given each hypothesis
for f = 1:nfeature
    for hh = 1:nkeephyp
        h = keephyp(hh);
        lambdaCond(hh,f) = lambdaT{f}(h);
        nCond(hh,f) = nT{f}(h);
    end
    
    pred(:,f) = poissonpdf(input(f), lambdaCond(:,f))*scale;
end

% Put predictions back into array with prediction=0 for belief=0 hypotheses
condSS.lambda = zeros(nhyp,nfeature);
condSS.lambda(keephyp,:) = lambdaCond;
condSS.n = zeros(nhyp,nfeature);
condSS.n(keephyp,:) = nCond;
tmp = pred;
pred = zeros(nhyp,nfeature);
pred(keephyp,:) = tmp;

% Prob ceiling at 1 (in case of variance << 1)
if any(pred > 1)
    error('A predictive prob is greater than one. Decrease predscale to combat this.');
end

% Check predictive probabilities
if any(isnan(pred) | ~isreal(pred))
    warning('ERROR with predictive probabilities')
    keyboard;
end

p = [];
p.prob = pred;
beliefs = beliefs(1:length(condSS.lambda))';
% p.expected = beliefs * condSS.lambda;
% p.error = abs(p.expected - obs);
p.ss = condSS;

end


% ==== UPDATE sufficient statistics with new observation ==============
function [cond_obs, suffstat] = update_POISSON(obs, cond_obs, suffstat, beliefs, prior)
% If prior==[], only update statistics.

nfeature = length(suffstat.n);
nhyp = sum(~isnan(suffstat.n{1}));
memory = length(suffstat.n{1});

% Skip update for any hyps with belief=0
keephyp = find(beliefs > 1e-300);
nkeephyp = length(keephyp);

% Replace NaNs with 0s to marginalize over missing context
obs_w_context = [cond_obs; obs];
obs_w_context(isnan(obs_w_context)) = 0;

for f = 1:nfeature
    
    % Update statistics, unless input obs is empty/missing
    if ~any(isnan(obs) | isempty(obs))
        
        new_lambda = sum(obs_w_context(:,f));
        
        n_update = suffstat.n{f}(keephyp) + 1;
        lambda_update = (suffstat.n{f}(keephyp).*suffstat.lambda{f}(keephyp) + repmat(new_lambda,nkeephyp,1))./n_update;
        
        suffstat.lambda{f}(keephyp) = lambda_update;
        suffstat.n{f}(keephyp) = n_update;
        
        % clear suffstats for hyps with beliefs=0
        suffstat.lambda{f}(~ismember(1:nhyp,keephyp)) = 0;
        suffstat.n{f}(~ismember(1:nhyp,keephyp)) = 0;
    end
    
    % Concatenating new hypothesis
    if ~isempty(prior)
        if nhyp < memory
            % add prior as newest hypothesis
            suffstat.n{f}(nhyp+1) = prior.n{f};
            suffstat.lambda{f}(nhyp+1) = prior.lambda{f};
        else
            % remove oldest hypothesis and add prior as newest hypothesis
            suffstat.n{f} = cat(1,suffstat.n{f}(2:end),prior.n{f});
            suffstat.lambda{f} = cat(2,suffstat.lambda{f}(2:end), prior.lambda{f});
        end
    end
end

% increment context to include new observation
cond_obs = [cond_obs; obs];
cond_obs(1,:) = [];

end


% ====== PDF functions =============================================
function p = studentpdf(x, mu, var, n)
c = exp(gammaln(n/2 + 0.5) - gammaln(n/2)) .* (n.*pi.*var).^(-0.5);
p = c .* (1 + (1./(n.*var)).*(x-mu).^2).^(-(n+1)/2);
end

function p = poissonpdf(x, lambda)
if abs(x - round(x)) > 1e-1
    error('Poisson PDF input x must be an integer.');
else
    x = round(x);
end
p = ((lambda.^x) / factorial(x)) .* exp(-lambda);
end
