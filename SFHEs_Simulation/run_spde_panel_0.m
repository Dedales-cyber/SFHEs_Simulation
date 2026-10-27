function results = run_spde_fd_threepanel(cfg, source, destination)


if nargin >= 1 && (ischar(cfg) || isa(cfg,'string'))
    if nargin < 2, source = []; end
    if nargin < 3, destination = []; end
    switch lower(char(cfg))
        case 'replot'
            results = replot_saved_results(source,destination);
        case 'export'
            assert(~isempty(source) && isscalar(source) && ...
                ishghandle(source) && strcmp(get(source,'Type'),'figure'), ...
                'The second argument must be a figure handle returned by openfig or gcf.');
            if isempty(destination)
                destination = fullfile(pwd,'results_edited');
            end
            folder = absolute_output_folder(destination);
            results = export_figure_files(source,folder,'Figure_1');
        otherwise
            error('Use a configuration structure, or mode ''replot'' / ''export''.');
    end
    return;
end
assert(nargin <= 1, 'Normal simulation accepts one configuration structure only.');


if nargin < 1, cfg = struct(); end
defaults = struct('theta',0.8, 'alpha',1.9, 'sigma',1, 'T',1, ...
    'interval',[0 1], 'x0',0.5, 'initialAmplitude',0, ...
    'nSpace',2500, 'R',500, 'M',[12000 1800000], ...
    'mFine',3600000, 'timeScheme','semiimplicit', ...
    'delta',[0.02 0.06 0.09 0.12 0.15 0.18 0.22 0.25], ...
    'histDelta',0.15, 'seed',20261021, 'batchSize',4, ...
    'maxMemoryMB',512, 'kernelQuadrature',2048, ...
    'coefficientTolerance',1e-12, ...
    'outputDir',fullfile(fileparts(mfilename('fullpath')), ...
                         'results_fd_em_threepanel_'));
assert(isstruct(cfg) && isscalar(cfg), 'cfg must be a scalar structure.');
names = fieldnames(defaults);
assert(all(ismember(fieldnames(cfg),names)), 'Unknown configuration field.');
for ii = 1:numel(names)
    if ~isfield(cfg,names{ii}), cfg.(names{ii}) = defaults.(names{ii}); end
end
cfg.delta = sort(unique(cfg.delta(:)'));
cfg.M = sort(cfg.M(:)');
cfg.timeScheme = lower(char(cfg.timeScheme));
validate_configuration(cfg);
cfg.outputDir = absolute_output_folder(cfg.outputDir);


n = cfg.nSpace;
J = n-1;                         
ell = diff(cfg.interval);
h = ell/n;
j = (1:J)';

lambdaH = (4/h^2)*sin(j*pi/(2*n)).^2;

muH = lambdaH.^(cfg.alpha/2);
muContinuous = (j*pi/ell).^cfg.alpha; 


[rhoObs,noiseSDObs,timeInfo] = euler_observation_transition(cfg,muH);


initial = zeros(J,1);
initial(1) = cfg.initialAmplitude*sqrt(ell/2);


v = ones(J,1);
Lh = spdiags([-v 2*v -v],-1:1,J,J)/h^2;
checkModes = unique([1 min(2,J) min(5,J) J]);
Qcheck = sqrt(2/n)*sin(((1:J)'*checkModes)*pi/n);
residual = Lh*Qcheck-bsxfun(@times,Qcheck,lambdaH(checkModes)');
spectrumResidual = norm(residual,'fro')/max(norm(Lh*Qcheck,'fro'),eps);
assert(spectrumResidual < 1e-10, 'FD eigenvalue/eigenvector check failed.');


[z,w] = gauss_legendre(cfg.kernelQuadrature);
normalizer = sqrt(sum(w.*bump(z).^2));
kappaFull = kernel_projections(cfg,J,z,w,normalizer);

[z2,w2] = gauss_legendre(2*cfg.kernelQuadrature);
normalizer2 = sqrt(sum(w2.*bump(z2).^2));
kappaCheck = kernel_projections(cfg,J,z2,w2,normalizer2);
quadratureGap = sqrt(sum((kappaFull-kappaCheck).^2,1))./ ...
    max(sqrt(sum(kappaCheck.^2,1)),eps);
if max(quadratureGap) > 1e-8
    error('Increase cfg.kernelQuadrature: kernel quadrature is unresolved.');
end


cutoff = cfg.coefficientTolerance*max(abs(kappaFull),[],1);
kappa = kappaFull;
kappa(bsxfun(@le,abs(kappa),cutoff)) = 0;
fullEnergy = sum(bsxfun(@times,muH,kappaFull.^2),1);
omittedEnergy = sum(bsxfun(@times,muH,(kappaFull-kappa).^2),1);
omittedEnergyFraction = omittedEnergy./max(fullEnergy,eps);
assert(max(omittedEnergyFraction) < 1e-12, ...
    'Reduce coefficientTolerance: omitted observation energy is too large.');
kernelMassGap = abs(1-sum(kappa.^2,1));

continuousEnergy = sum(bsxfun(@times,muContinuous,kappaFull.^2),1);
fdEnergyGap = abs(fullEnergy-continuousEnergy)./max(continuousEnergy,eps);
if max(kernelMassGap) > 1e-6 || max(fdEnergyGap) > 1e-3
    warning('SPDE:SpaceResolution', ...
        ['Spatial kernel diagnostics are not small. Increase nSpace and ' ...
         'check again; this diagnostic does not certify estimator convergence.']);
end



psi = full_space_psi(cfg.alpha,normalizer,256,1/1024);
psiCheck = full_space_psi(cfg.alpha,normalizer,512,1/1024);


psiGap = abs(psi-psiCheck)/psiCheck;
assert(psiGap < 1e-5, 'Increase the Fourier window for Psi.');


psi = cfg.sigma^2 * psiCheck;


theorySD = cfg.sigma * cfg.delta.^(cfg.alpha/2) ...
    * sqrt(cfg.theta/(cfg.T*psi));

cAlpha = sqrt(cfg.T*psi/cfg.theta)/cfg.sigma;


fprintf('FD model: n=%d subintervals, J=%d modes, h=%.8g\n',n,J,h);
fprintf('M=[%d %d], R=%d, histogram delta=%.4g\n', ...
    cfg.M(1),cfg.M(2),cfg.R,cfg.histDelta);
fprintf('Scheme=%s, mFine=%d, dtFine=%.12g, fine strides=[%d %d]\n', ...
    cfg.timeScheme,cfg.mFine,timeInfo.dtFine, ...
    cfg.mFine/cfg.M(1),cfg.mFine/cfg.M(2));
fprintf('Explicit-only stability number theta*muMax*dtFine=%.6g; threshold=2.\n', ...
    timeInfo.stabilityNumber);
if strcmp(cfg.timeScheme,'semiimplicit')
    fprintf('Semiimplicit EM is used; the explicit threshold does not apply.\n');
end
fprintf('Psi=%.12g; FD spectrum residual=%.3g\n',psi,spectrumResidual);
fprintf('cAlpha=%.12g; n^alpha=%.12g\n',cAlpha,n^cfg.alpha);
fprintf('Max kernel mass gap=%.3g; FD/continuous energy gap=%.3g\n', ...
    max(kernelMassGap),max(fdEnergyGap));
fprintf('Max skipped energy fraction=%.3g; active observed modes=%d/%d\n', ...
    max(omittedEnergyFraction),sum(any(kappa~=0,2)),J);
fprintf('The full run may take hours; all delta values share modal paths.\n');


thetaHat = simulate_estimates(cfg,muH,kappa,initial,rhoObs,noiseSDObs);
D = numel(cfg.delta);
rmse = zeros(D,2);
slope = zeros(1,2);
standardized = zeros(size(thetaHat));
rows = zeros(2*D,16);

for mm = 1:2
    
    standardized(:,:,mm) = bsxfun(@rdivide, ...
        thetaHat(:,:,mm)-cfg.theta,theorySD);
    for dd = 1:D
        err = thetaHat(:,dd,mm)-cfg.theta;
        U = standardized(:,dd,mm);
        
        rmse(dd,mm) = sqrt(mean(err.^2));
        rows((mm-1)*D+dd,:) = [cfg.delta(dd),cfg.M(mm), ...
            cfg.delta(dd)^(-2*cfg.alpha)/cfg.M(mm), ...
            n,J,h,cfg.R,mean(err),std(thetaHat(:,dd,mm),0), ...
            theorySD(dd),rmse(dd,mm),mean(U),std(U,0), ...
            mean(abs(U)<=1.959963984540054),kernelMassGap(dd),fdEnergyGap(dd)];
    end
    
    p = polyfit(log(cfg.delta(:)),log(rmse(:,mm)),1);
    slope(mm) = p(1);
end
summary = array2table(rows,'VariableNames', ...
    {'Delta','M','CLTRatio','SpaceSubintervals','Modes','H','R', ...
     'Bias','EmpiricalSD','TheorySD','RMSE','StandardizedMean', ...
     'StandardizedSD','Coverage95','KernelMassGap','FDEnergyRelativeGap'});
summary.FineSteps = repmat(cfg.mFine,height(summary),1);
summary.FineStep = repmat(timeInfo.dtFine,height(summary),1);
summary.TimeScheme = repmat({cfg.timeScheme},height(summary),1);

slopeTable = table(cfg.M(:),slope(:),repmat(cfg.alpha/2,2,1), ...
    'VariableNames',{'M','FittedSlope','ReferenceSlope'});
histIndex = find(abs(cfg.delta-cfg.histDelta)<1e-12,1);

Uhist = reshape(standardized(:,histIndex,:),cfg.R,2);


results = struct('config',cfg,'nModes',J,'h',h,'lambdaH',lambdaH, ...
    'muH',muH,'kappa',kappa,'initial',initial,'Psi',psi,'cAlpha',cAlpha, ...
    'timeInfo',timeInfo, ...
    'thetaHat',thetaHat,'RMSE',rmse,'fittedSlope',slope, ...
    'theorySD',theorySD,'standardized',standardized,'summary',summary, ...
    'slopeTable',slopeTable,'quadratureGap',quadratureGap, ...
    'omittedEnergyFraction',omittedEnergyFraction);

results = save_results_and_figure(results,Uhist);
end

function estimates = simulate_estimates(cfg,mu,kappa,initial,rho,noiseSD)

Mstore = cfg.M(2);
D = numel(cfg.delta);

bytesPerPath = 8*(Mstore+1)*(2*D+6);
memoryBatch = floor(cfg.maxMemoryMB*1024^2/bytesPerPath);
assert(memoryBatch >= 1, 'Increase maxMemoryMB or reduce M/delta count.');
Bmax = min([cfg.batchSize cfg.R memoryBatch]);
active = find(any(kappa~=0,2));
estimates = zeros(cfg.R,D,2);

rng(cfg.seed,'twister');
for first = 1:Bmax:cfg.R
    last = min(first+Bmax-1,cfg.R);
    B = last-first+1;
    Y = zeros(Mstore+1,B,D);
    Z = zeros(Mstore+1,B,D);
    for kk = 1:numel(active)
        k = active(kk);
        
        xi = randn(Mstore,B);
       
        u = filter(noiseSD(k),[1 -rho(k)],xi, ...
            rho(k)*initial(k)*ones(1,B),1);
        clear xi;
        targets = find(kappa(k,:)~=0);
        for dd = targets(:)'
            w = kappa(k,dd);
            
            Y(1,:,dd) = Y(1,:,dd)+w*initial(k);
            Z(1,:,dd) = Z(1,:,dd)-mu(k)*w*initial(k);
            Y(2:end,:,dd) = Y(2:end,:,dd)+w*u;
            Z(2:end,:,dd) = Z(2:end,:,dd)-mu(k)*w*u;
        end
        clear u;
    end
    for mm = 1:2
        
        stride = Mstore/cfg.M(mm);
        index = 1:stride:Mstore+1;
        dtObs = cfg.T/cfg.M(mm);
        for dd = 1:D
            Yo = Y(index,:,dd);
            Zo = Z(index,:,dd);
            
            numerator = sum(Zo(1:end-1,:).*diff(Yo,1,1),1);
            denominator = dtObs*sum(Zo(1:end-1,:).^2,1);
            assert(all(isfinite(denominator) & denominator>0), ...
                'Invalid denominator: inspect model and resolution.');
            estimate = numerator./denominator;
            
            assert(all(isfinite(estimate)), 'Nonfinite estimator.');
            estimates(first:last,dd,mm) = estimate';
        end
    end
    clear Y Z Yo Zo;
    fprintf('Completed %d/%d Monte Carlo paths (batch=%d).\n',last,cfg.R,B);
    drawnow;
end
end

function fig = draw_two_panels(cfg,rmse,slope,theorySD,U)

assert(size(U,1)==cfg.R && size(U,2)==2, 'Plot sample size does not match cfg.R.');
assert(all(isfinite(rmse(:)) & rmse(:)>0) && all(theorySD(:)>0), ...
    'RMSE and theoretical SD must be positive before taking log10.');


colors   = [0.051 0.251 0.749;   
            0.749 0.149 0.149];  
markers = {'o','s'};
histFace = [0.302 0.490 0.780;   
            0.820 0.458 0.458];  
histEdge = [0.051 0.251 0.749;   
            0.749 0.149 0.149];  
histAlpha = 0.5;

estimatorNames = cell(1,2);
for mm = 1:2
    
    estimatorNames{mm} = sprintf('$\\widehat{\\vartheta}_{M_%d,\\delta}$',mm);
end
fig = figure('Color','w','Units','centimeters','Position',[1 1 24 9], ...
    'Name','RMSE and histogram diagnostics','NumberTitle','off');
positions = [0.080 0.16 0.440 0.73; ...
             0.620 0.16 0.350 0.73];
ax = gobjects(1,2);
for pp = 1:2
    ax(pp) = axes('Parent',fig,'Position',positions(pp,:));
    hold(ax(pp),'on'); box(ax(pp),'on'); grid(ax(pp),'on');
    set(ax(pp),'FontName','Times New Roman','FontSize',11.5,'LineWidth',0.9, ...
        'XScale','linear','YScale','linear','TickLabelInterpreter','tex');
end


x = log10(cfg.delta(:));
xFit = linspace(min(x),max(x),300)';
handles = gobjects(1,5);
for mm = 1:2
    y = log10(rmse(:,mm));
    
    handles(2*mm-1) = plot(ax(1),x,y,['-' markers{mm}], ...
        'Color',colors(mm,:),'MarkerFaceColor',colors(mm,:), ...
        'LineWidth',1.6,'MarkerSize',5.5,'DisplayName',estimatorNames{mm});
    
    intercept10 = mean(y)-slope(mm)*mean(x);
    yFit = intercept10+slope(mm)*xFit;
    
    fitName = sprintf('fitted slope $%.3f$ (%s)', ...
    slope(mm), estimatorNames{mm});
    handles(2*mm) = plot(ax(1),xFit,yFit,'--', ...
        'Color',colors(mm,:),'LineWidth',1.35,'DisplayName',fitName);
end
handles(5) = plot(ax(1),x,log10(theorySD(:)),'k:', ...
    'LineWidth',1.7,'DisplayName', ...
    sprintf('reference slope $%.2f$',cfg.alpha/2));
xlabel(ax(1),'$\log_{10}(\delta)$','Interpreter','latex','FontSize',12.5);
ylabel(ax(1),'$\log_{10}(\mathrm{RMSE})$','Interpreter','latex','FontSize',12.5);
title(ax(1),'$\log_{10}$-$\log_{10}$ plot of RMSEs', ...
    'Interpreter','latex','FontSize',11);

legend(ax(1),handles,'Location','northwest', ...
    'FontSize',8,'Interpreter','latex','Box','on');
set(ax(1),'Position',positions(1,:));

plotY = [];
for kk = 1:numel(handles)
    values = get(handles(kk),'YData');
    plotY = [plotY;values(:)]; 
end
spanY = max(plotY)-min(plotY);
if spanY>0
    ylim(ax(1),[min(plotY)-0.06*spanY, max(plotY)+0.65*spanY]);
end


lo = min(-4.2,min(U(:))-0.2);
hi = max(4.2,max(U(:))+0.2);
edges = linspace(lo,hi,31);
for mm = 1:2
    histogram(ax(2),U(:,mm),edges,'Normalization','pdf', ...
        'FaceColor',histFace(mm,:),'EdgeColor',histEdge(mm,:), ...
        'FaceAlpha',histAlpha,'LineWidth',1.0, ...
                'DisplayName',sprintf('$c^{(\\alpha)}\\delta^{-\\alpha/2}(\\widehat{\\vartheta}_{M_%d,\\delta}-\\vartheta)$',mm));

end
zz = linspace(lo,hi,600);

plot(ax(2),zz,exp(-zz.^2/2)/sqrt(2*pi),'k:','LineWidth',1.7, ...
    'DisplayName','$N(0,1)$');
xlim(ax(2),[lo hi]);

xlabel(ax(2),'');
ylabel(ax(2),'');
title(ax(2),sprintf('Histograms ($\\delta=%.3g$)',cfg.histDelta), ...
    'Interpreter','latex','FontSize',12);
legend(ax(2),'Location','northwest','FontSize',8,'Interpreter','latex','Box','on');
set(ax(2),'Position',positions(2,:));


drawnow;
end

function results = replot_saved_results(matFile,outputFolder)

if isempty(matFile)
    [name,folder] = uigetfile('*.mat','Select fd_threepanel_results.mat');
    if isequal(name,0), results = []; return; end
    matFile = fullfile(folder,name);
end
matFile = char(matFile);
assert(exist(matFile,'file')==2, 'The selected MAT file does not exist.');
loaded = load(matFile,'results');
assert(isfield(loaded,'results') && isstruct(loaded.results), ...
    'The MAT file must contain the results structure saved by this program.');
results = loaded.results;
needed = {'config','thetaHat','RMSE','fittedSlope','theorySD', ...
          'standardized','summary','slopeTable','Psi','cAlpha','nModes'};
assert(all(isfield(results,needed)), 'The saved results structure is incomplete.');
cfg = results.config;
assert(size(results.thetaHat,1)==cfg.R, 'Saved sample size differs from config.R.');
histIndex = find(abs(cfg.delta-cfg.histDelta)<1e-12,1);
assert(~isempty(histIndex), 'Saved histDelta does not occur in delta.');
Uhist = reshape(results.standardized(:,histIndex,:),cfg.R,2);
if isempty(outputFolder)
    [folder,~,~] = fileparts(matFile);
    if isempty(folder), folder = pwd; end
    outputFolder = fullfile(folder,'replotted');
end
results.config.outputDir = absolute_output_folder(outputFolder);
fprintf('Replotting saved samples only: R=%d. No new Monte Carlo paths.\n',cfg.R);
results = save_results_and_figure(results,Uhist);
end

function results = save_results_and_figure(results,Uhist)

cfg = results.config;
folder = cfg.outputDir;
results.imageFiles = struct('pdf',fullfile(folder,'Figure_1.pdf'), ...
    'png',fullfile(folder,'Figure_1.png'),'fig',fullfile(folder,'Figure_1.fig'));
writetable(results.summary,fullfile(folder,'fd_threepanel_statistics.csv'));
writetable(results.slopeTable,fullfile(folder,'fd_threepanel_slopes.csv'));
save(fullfile(folder,'fd_threepanel_results.mat'),'results','-v7.3');
fid = fopen(fullfile(folder,'numerical_constants.txt'),'w');
assert(fid>=0, 'Cannot write numerical_constants.txt.');
guard = onCleanup(@() fclose(fid));
fprintf(fid,'Psi = %.12g\ncAlpha = %.12g\n',results.Psi,results.cAlpha);
fprintf(fid,'n = %d\nJ = %d\nR = %d\nmFine = %d\ntimeScheme = %s\n', ...
    cfg.nSpace,results.nModes,cfg.R,cfg.mFine,cfg.timeScheme);
clear guard;
write_plot_notes(results);
fprintf('Numerical results saved before plotting:\n%s\n', ...
    fullfile(folder,'fd_threepanel_results.mat'));
fig = draw_two_panels(cfg,results.RMSE,results.fittedSlope,results.theorySD,Uhist);
export_figure_files(fig,folder,'Figure_1');
disp(results.slopeTable);
end

function folder = absolute_output_folder(folder)

folder = char(folder);
assert(~isempty(folder), 'The output folder cannot be empty.');
if ~exist(folder,'dir')
    [ok,message] = mkdir(folder);
    assert(ok, 'Cannot create output folder: %s',message);
end
[ok,attributes] = fileattrib(folder);
assert(ok, 'Cannot resolve the output folder.');
folder = attributes.Name;
end

function files = export_figure_files(fig,folder,stem)

folder = absolute_output_folder(folder);
files = struct('pdf',fullfile(folder,[stem '.pdf']), ...
    'png',fullfile(folder,[stem '.png']),'fig',fullfile(folder,[stem '.fig']));
temporaryPDF = [tempname(folder) '.pdf'];
guard = onCleanup(@() remove_file_if_exists(temporaryPDF));
firstMessage = 'exportgraphics is unavailable in this MATLAB version.';
pdfOK = false;
drawnow;
if exist('exportgraphics','file')~=0
    try
        exportgraphics(fig,temporaryPDF,'ContentType','vector','BackgroundColor','white');
        pdfOK = valid_pdf_file(temporaryPDF);
        if ~pdfOK, firstMessage = 'exportgraphics did not produce a valid PDF file.'; end
    catch firstError
        firstMessage = firstError.message;
    end
end
if ~pdfOK
    remove_file_if_exists(temporaryPDF);
    fprintf('Trying print as PDF fallback. First method: %s\n',firstMessage);
    configure_print_page(fig);
    try
        if ~verLessThan('matlab','9.11')
            
            print(fig,temporaryPDF,'-dpdf','-vector');
        else
            print(fig,temporaryPDF,'-dpdf','-painters');
        end
        pdfOK = valid_pdf_file(temporaryPDF);
        if ~pdfOK, error('The fallback output is missing or is not a PDF.'); end
    catch secondError
        error('SPDE:PDFExportFailed', ...
            ['PDF export failed. Target: %s\nFirst method: %s\n' ...
             'Fallback: %s\nThe saved numerical results can be replotted without simulation.'], ...
            files.pdf,firstMessage,secondError.message);
    end
end
[ok,message] = movefile(temporaryPDF,files.pdf,'f');
assert(ok, 'Cannot place PDF at %s: %s',files.pdf,message);
clear guard;
assert(valid_pdf_file(files.pdf), 'The final PDF could not be verified.');
information = dir(files.pdf);
fprintf('PDF saved and verified (%d bytes):\n%s\n',information.bytes,files.pdf);


try
    savefig(fig,files.fig);
    fprintf('Editable FIG:\n%s\n',files.fig);
catch figureError
    warning('SPDE:FIGExportFailed','PDF is saved, but FIG export failed: %s',figureError.message);
end
pngOK = false;
if exist('exportgraphics','file')~=0
    try
        exportgraphics(fig,files.png,'Resolution',300,'BackgroundColor','white');
        pngOK = true;
    catch
        
    end
end
if ~pngOK
    try
        configure_print_page(fig);
        print(fig,files.png,'-dpng','-r300');
        pngOK = true;
    catch pngError
        warning('SPDE:PNGExportFailed','PDF is saved, but PNG export failed: %s',pngError.message);
    end
end
if pngOK, fprintf('PNG:\n%s\n',files.png); end
end

function configure_print_page(fig)

oldUnits = get(fig,'Units');
set(fig,'Units','inches');
position = get(fig,'Position');
set(fig,'Units',oldUnits);
width = position(3);
height = position(4);
set(fig,'PaperUnits','inches','PaperSize',[width height], ...
    'PaperPosition',[0 0 width height],'PaperPositionMode','manual', ...
    'InvertHardcopy','off');
end

function ok = valid_pdf_file(path)

ok = false;
if exist(path,'file')~=2, return; end
information = dir(path);
if isempty(information) || information.bytes < 100, return; end
fid = fopen(path,'rb');
if fid<0, return; end
guard = onCleanup(@() fclose(fid));
header = fread(fid,5,'*char')';
ok = strcmp(header,'%PDF-');
end

function remove_file_if_exists(path)

if exist(path,'file')==2
    try, delete(path); catch, end
end
end

function write_plot_notes(results)

cfg = results.config;
fid = fopen(fullfile(cfg.outputDir,'figure_notes.txt'),'w');
assert(fid>=0, 'Cannot write figure_notes.txt.');
guard = onCleanup(@() fclose(fid));
fprintf(fid,'Estimator legend notation: vartheta_hat_{delta,M_k}. True parameter: vartheta.\n');
fprintf(fid,'R = %d independent Monte Carlo paths.\n',cfg.R);
fprintf(fid,'M1 = %d; M2 = %d observation intervals; each grid has M+1 time points.\n',cfg.M);
fprintf(fid,'delta values:'); fprintf(fid,' %.12g',cfg.delta); fprintf(fid,'\n');
fprintf(fid,'Histogram delta = %.12g; each group contains R samples.\n',cfg.histDelta);
fprintf(fid,'U=(vartheta_hat-vartheta)/s_delta; centering uses the true parameter.\n');
fprintf(fid,'Psi=(sigma^2/2)*<(-Delta_R)^(alpha/2)K,K>.\n');
fprintf(fid,'s_delta=sigma*delta^(alpha/2)*sqrt(vartheta/(T*Psi)); theoretical asymptotic SD.\n');
fprintf(fid,'cAlpha=sqrt(T*Psi/vartheta)/sigma; ||K||_L2=1.\n');
fprintf(fid,'The empirical SD is NOT used to standardize the histograms.\n');
fprintf(fid,'Psi = %.12g; cAlpha = %.12g.\n',results.Psi,results.cAlpha);
fprintf(fid,'Reference: asymptotic SD, log-log slope alpha/2 = %.12g.\n',cfg.alpha/2);
fprintf(fid,'Fit: least squares log(RMSE)=intercept+slope*log(delta), using ALL listed deltas.\n');
fprintf(fid,'Fitted slopes: M1 %.12g; M2 %.12g.\n',results.fittedSlope);
fprintf(fid,'Two fitted lines are drawn over the displayed delta range; slopes are descriptive.\n');


fprintf(fid,'Two panels only: RMSE/fits and histograms. No QQ plot.\n');
fprintf(fid,'Left axes: x=log10(delta), y=log10(RMSE); both axes use LINEAR ticks.\n');
fprintf(fid,'Legend estimator subscripts: delta,M_k. Title subscripts: M,delta, as requested.\n');
fprintf(fid,'Right title: Histogram, (delta=histDelta); axis names are empty, numeric ticks retained.\n');
fprintf(fid,'Both legends are inside the upper-left corner, font size 8; no bottom M1/M2/R annotation.\n');
fprintf(fid,'Fixed M experiments do not themselves form a joint sequence q_delta -> 0.\n');
end

function kappa = kernel_projections(cfg,J,z,w,normalizer)

ell = diff(cfg.interval);
frequency = (1:J)'*pi/ell;
phase = sin(frequency*(cfg.x0-cfg.interval(1)));

if abs(cfg.x0-mean(cfg.interval)) < 10*eps(max(1,ell))
    phase(2:2:end) = 0;
end
weightedKernel = w.*bump(z)/normalizer;
kappa = zeros(J,numel(cfg.delta));
for dd = 1:numel(cfg.delta)
    d = cfg.delta(dd);
    kappa(:,dd) = sqrt(2*d/ell)*phase.* ...
        (cos((frequency*d)*z')*weightedKernel);
end
end

function psi = full_space_psi(alpha,normalizer,L,dx)

N = round(L/dx);
assert(mod(N,2)==0, 'Use an even FFT length.');
x = ((0:N-1)'-N/2)*dx;
K = bump(x)/normalizer;
omega = (2*pi/L)*(-N/2:N/2-1)';
F2 = abs(fftshift(fft(K))).^2;
psi = 0.5*dx/N*sum(abs(omega).^alpha.*F2);
assert(isfinite(psi) && psi>0, 'Invalid theoretical kernel energy.');
end

function y = bump(z)

y = zeros(size(z));
inside = abs(z)<1;
y(inside) = exp(-10./(1-z(inside).^2));
end

function [z,w] = gauss_legendre(N)

half = ceil(N/2);
k = (1:half)';
x = cos(pi*(k-0.25)/(N+0.5));
for iter = 1:30
    [PN,PNminus] = legendre_pair(N,x);
    derivative = N*(x.*PN-PNminus)./(x.^2-1);
    correction = PN./derivative;
    x = x-correction;
    if max(abs(correction))<5e-15, break; end
end
assert(iter<30,'Gauss-Legendre Newton iteration failed.');
[PN,PNminus] = legendre_pair(N,x);
derivative = N*(x.*PN-PNminus)./(x.^2-1);
weight = 2./((1-x.^2).*derivative.^2);
if mod(N,2)==0
    z = [-x;flipud(x)];
    w = [weight;flipud(weight)];
else
    x(end) = 0;
    z = [-x;flipud(x(1:end-1))];
    w = [weight;flipud(weight(1:end-1))];
end
end

function [PN,PNminus] = legendre_pair(N,x)

P0 = ones(size(x));
P1 = x;
for k = 2:N
    P2 = ((2*k-1)*x.*P1-(k-1)*P0)/k;
    P0 = P1;
    P1 = P2;
end
PN = P1;
PNminus = P0;
end

function [rhoObs,sdObs,info] = euler_observation_transition(cfg,mu)

dt = cfg.T/cfg.mFine;
rate = cfg.theta*mu;
number = max(rate)*dt;
switch cfg.timeScheme
    case 'semiimplicit'
        r = 1./(1+rate*dt);
        b = cfg.sigma*sqrt(dt)*r;
    case 'explicit'
        r = 1-rate*dt;
        b = cfg.sigma*sqrt(dt)*ones(size(rate));
        if any(abs(r)>=1)
            error('SPDE:ExplicitUnstable', ...
                ['Explicit EM is unstable: theta*muMax*dt=%.6g >=2. ' ...
                 'Use semiimplicit, or mFine>%.6g with both M dividing mFine.'], ...
                number,cfg.T*max(rate)/2);
        end
end
s = cfg.mFine/cfg.M(2);
rhoObs = r.^s;

logR2 = 2*log(abs(r));
geometric = expm1(s*logR2)./expm1(logR2);
geometric(r.^2==1) = s;
sdObs = b.*sqrt(geometric);
assert(all(isfinite(sdObs) & sdObs>0),'Invalid Euler innovation variance.');
info = struct('dtFine',dt,'fineSteps',cfg.mFine, ...
    'stepsPerStoredObservation',s,'stabilityNumber',number, ...
    'scheme',cfg.timeScheme,'fineRho',r,'fineNoiseSD',b);
end

function validate_configuration(cfg)

assert(cfg.alpha>1 && cfg.alpha<=2 && cfg.theta>0 && cfg.sigma>0 && cfg.T>0, ...
    'Require 1<alpha<=2 and positive theta, sigma, T.');
assert(numel(cfg.interval)==2 && diff(cfg.interval)>0, 'Invalid interval.');
assert(cfg.nSpace>=4 && cfg.nSpace==round(cfg.nSpace), ...
    'nSpace must be an integer >=4.');
assert(cfg.R>=3 && cfg.R==round(cfg.R), 'R must be an integer >=3.');
assert(numel(cfg.M)==2 && all(cfg.M>=2) && all(cfg.M==round(cfg.M)) ...
    && cfg.M(1)<cfg.M(2), 'M must contain two distinct positive integers >=2.');
assert(mod(cfg.M(2),cfg.M(1))==0, ...
    'For nested sampling, the larger M must be a multiple of the smaller M.');
assert(cfg.mFine>=cfg.M(2) && cfg.mFine==round(cfg.mFine) ...
    && all(mod(cfg.mFine,cfg.M)==0), ...
    'mFine must be an integer multiple of both observation counts M.');
assert(ismember(cfg.timeScheme,{'semiimplicit','explicit'}), ...
    'timeScheme must be semiimplicit or explicit.');
assert(numel(cfg.delta)>=3 && all(cfg.delta>0), 'Use at least 3 positive deltas.');
assert(any(abs(cfg.delta-cfg.histDelta)<1e-12), ...
    'histDelta must be one of the delta values.');
assert(cfg.x0-max(cfg.delta)>cfg.interval(1) ...
    && cfg.x0+max(cfg.delta)<cfg.interval(2), ...
    'Every kernel support must stay strictly inside the interval.');
assert(cfg.batchSize>=1 && cfg.batchSize==round(cfg.batchSize) ...
    && cfg.maxMemoryMB>0, 'Invalid batchSize/maxMemoryMB.');
assert(cfg.kernelQuadrature>=32 && cfg.kernelQuadrature==round(cfg.kernelQuadrature), ...
    'kernelQuadrature must be an integer >=32.');
assert(cfg.coefficientTolerance>=0 && cfg.coefficientTolerance<1, ...
    'coefficientTolerance must be in [0,1).');
end
