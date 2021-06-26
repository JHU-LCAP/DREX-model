function display_DREX_output(mdl_out, feat_in)
% Usage: display_DREX_output(mdl, x)
% Helper function for displaying D-REX Model output
%
% Input Arguments:
%   mdl_out              output from run_DREX_model.m
%   feat_in              input sequence of observations
%
% v2
% Benjamin Skerritt-Davis
% bsd@jhu.edu
% 6/25/2019

clf;

if sum(size(feat_in)>1)==1 % single feature observations
    feat_in = reshape(feat_in,[],1);
elseif size(feat_in,1) < size(feat_in,2) % assume more time-pts than features
    error('x dim should be: time x numFeatures');
end
numFeatures = size(feat_in,2);

time = 1:length(feat_in);

col = lines(numFeatures);


subplot(3,1,1);
% cool colormap for prediction
cmap = zeros(256,3);
cmap(:,1) = linspace(1,0.4,256)';
cmap(:,2) = linspace(1,0.4,256)';
cmap(:,3) = 1;
nlevel = 500;
PD = [];

for f = 1:numFeatures
    if ~strcmp(mdl_out.distribution,'poisson') 
        if f > 1
            yshift = yshift+ max(feat_in(:,f-1)) - min(feat_in(:,f)) + 0.5*range(feat_in(:,f));
        else
            yshift = 0;
        end
        hold all;
        if isfield(mdl_out, 'prediction_params')
            
            [PD,X,Y] = post_DREX_prediction(f,mdl_out,linspace(min(feat_in(:,f)) - 0.1*range(feat_in(:,f)),max(feat_in(:,f)) + 0.1*range(feat_in(:,f)),100));
            try
            contourf(X,Y+yshift,PD,nlevel,'fill','on','linestyle','-','linecolor','none');%0.6*[1 1 1])
            catch
                keyboard;
            end
            set(gca,'colormap',cmap);
        else
            Y = [min(feat_in) max(feat_in)];
        end
        prettyplot_sequence(feat_in(:,f),3,gcf,false,col(f,:)*0.5,yshift);
        hold off;
    else
        Y = [0 1];
        stem(feat_in,'marker','none','linewidth',1);
    end
end
hold off;
title('Observations')
xlim([1 length(feat_in)+1])
xlabel('Time');
xticks = get(gca,'XTick');
xticklabels = xticks;
% if range(Y(:))==0
%     ylim(mean(Y(:))+[-1 1]);
% else
%     ylim([min(Y(:)) max(Y(:))])
% end
if ~isempty(PD)
    caxis([0 quantile(PD(:),0.99)]);
end


subplot(3,1,2);
mdl_out.context_beliefs(mdl_out.context_beliefs==0) = nan;
sz = size(mdl_out.context_beliefs);

newcp = nan(sz(2)*[1 1]); % Bend context posterior back for display
newcp(1:sz(1),1:sz(1)) = mdl_out.context_beliefs(:,1:sz(1));
for t = 1+sz(1):sz(2)
    newcp(t-sz(1)+1:t,t) = mdl_out.context_beliefs(:,t);
end

p = pcolor(1+(0:sz(2)-1),1+(0:sz(2)-1),log10(newcp));
set(gca,'Color',0.95*ones(1,3),'colormap',parula);
xlim([1 size(feat_in,1)+1])
p.LineStyle = 'none';
axis xy;
title('P( c_i | x_{1:t} )')
caxis([-5 0])
ylabel('Context boundary')
xlabel('Time');
set(gca,'YTick',xticks,'YTickLabel',xticklabels);
grid on;


subplot(3,1,3)
for f = 1:numFeatures
    plot((1:size(feat_in,1)), mdl_out.surprisal(:,f),'-','color',col(f,:),'LineWidth',1);
    hold all;
    scatter((1:size(feat_in,1)), mdl_out.surprisal(:,f),20,col(f,:),'filled');
end
hold off;
xlim([0 size(feat_in,1)+1])
title('Surprisal')
xlabel('Time');
ylabel('Surprisal (bits)')


ax = findall(gcf,'Type','Axes');
for i = 1:length(ax)
    set(ax(i),'FontSize',12,'XTick',xticks)
    if nargin > 2
        set(ax(i),'XTickLabel',time(xticks));
    end
end


end


function [xpos, ypos] = prettyplot_sequence(x,width,fignum,connect,color,yshift)
% Usage: [xpos, ypos] = prettyplot_sequence(x,width,fignum,connect,color,yshift)

if nargin < 2
    width = 5;
end
if nargin < 3
    figure;
else
    figure(fignum);
end
if nargin < 4
    connect = true;
end
if nargin < 5
    color = [0,0,0];
end
if nargin < 6
    yshift = 0;
end
x = reshape(x,[],1);



xpos = reshape([1:length(x); (1:length(x))+.15/.175; nan(1,length(x))],1,[]);
ypos = reshape([x x nan(length(x),1)]',1,[]);
ypos = ypos+yshift;

plot(xpos, ypos, 'LineWidth', width, 'Color',color);

if connect
    hold all;
    xpos2 = reshape([(1:length(x)-1)+.15/.175; 2:length(x); nan(1,length(x)-1)],1,[]);
    ypos2 = reshape([x(1:end-1) x(2:end) nan(length(x)-1,1)]',1,[]);
    ypos2 = ypos2+yshift;
    plot(xpos2,ypos2,':k','LineWidth',1,'Color',color);
    hold off;
end
xlim([0 length(x)+1])
end

