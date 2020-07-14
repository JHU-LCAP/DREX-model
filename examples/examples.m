%% Simple example of Gaussian sequence with one change
rng(10);

x = [2*randn(50,1); 6*randn(50,1)];

params = [];
params.distribution = 'gaussian';
params.D = 1;
params.prior = estimate_suffstat(std(x)*randn(1000,1),params);

x = [2*randn(50,1); 6*randn(50,1)];
out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)

%% Same example w/ finite maxhyp
rng(10);
x = [2*randn(50,1); 6*randn(50,1)];

params = [];
params.distribution = 'gaussian';
params.D = 1;
params.prior = estimate_suffstat(std(x)*randn(1000,1),params);
params.maxhyp = 30; % limits number of context hypotheses to 30, pruning by beliefs

x = [2*randn(50,1); 6*randn(50,1)];
out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Gaussian mixture example with single wide component in prior

x = [-5+getFractalSequence(2,50), 5+getFractalSequence(2,50)]';
x = x(randperm(length(x)));
x = [rand(100,1)*range(x) + min(x); x];


params = [];
params.distribution = 'gmm';
params.max_ncomp = 10;
params.beta = 0.001;
params.D = 1;
params.prior = estimate_suffstat(randn(1000,1),params);
params.maxhyp = inf;
params.memory = inf;

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Gaussian mixture example with spread out multi-component prior

x = [-5+getFractalSequence(2,50), 5+getFractalSequence(2,50)]';
x = x(randperm(length(x)));
x = [rand(100,1)*range(x) + min(x); x];


params = [];
params.distribution = 'gmm';
params.max_ncomp = 10;
params.beta = 0.001;
params.D = 1;
params.prior = estimate_suffstat(randn(1000,1),params);
params.prior.mu = {[-8:2:8 nan]};
params.prior.sigma = {[0.1*ones(1,9) nan]};
params.prior.n = {[ones(1,9) nan]};
params.prior.pi = {[1/9*ones(1,9) 0]};
params.prior.sp = {[ones(1,9) 0]};
params.prior.k = {9};


params.maxhyp = inf;
params.memory = inf;

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Simple example w/ prior from several different sequences
exposure = cell(3,1);
exposure{1} = 2*randn(50,1);
exposure{2} = 6*randn(20,1);
exposure{3} = 0.5*randn(80,1);

x = [2*randn(50,1); 6*randn(50,1)];

params = [];
params.D = 1;
params.prior = estimate_suffstat(exposure,params); % priors calculated from exposure set

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Example with temporal dependence D = 2 
x = zeros(100,1); % input: Gaussian random walk
for t = 2:100
    if t < 50
        s = .2;
    else
        s = -.2;
    end
    x(t) = x(t-1)+2*(randn(1)+s);
end

params = [];
params.D = 2;
params.memory = inf;
params.prior = estimate_suffstat(x,params);

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Example with temporal dependence D >> 1 
exposure = 5*randn(100,1);
x = 5*repmat(randn(5,1),15,1)+randn(75,1); % noisy cycle

params = [];
params.D = 6;
params.memory = inf;
params.prior = estimate_suffstat(exposure,params);

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)

%% Example with finite memory
x = [2*randn(50,1); 6*randn(50,1)];

params = [];
params.D = 1;
params.memory = 30;
params.prior = estimate_suffstat(permute(x,[1,3,2]),params);

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Example with observation noise
x = [2*randn(50,1); 6*randn(50,1)];

params = [];
params.D = 1;
params.memory = inf;
params.obsnz = 3;
params.prior = estimate_suffstat(permute(x,[1,3,2]),params);

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Example with missing/silent observations
x = [2*randn(50,1); 6*randn(50,1)];
x([30:35, 70:80]) = nan;

params = [];
params.D = 1;
params.prior = estimate_suffstat(permute(x,[1,3,2]),params);

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)


%% Example with multiple features
x1 = [2*randn(70,1); 6*randn(30,1)];
x2 = [6*randn(50,1); 2*randn(50,1)];
x3 = [2*randn(30,1); 8+2*randn(70,1)];
x = [x1, x2, x3];

params = [];
params.D = 1;
params.prior = estimate_suffstat(permute(x,[1,3,2]),params);

out = run_DREX_model(x,params);

figure(1); clf;
display_DREX_output(out,x)
