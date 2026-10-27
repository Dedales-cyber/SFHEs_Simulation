function results = run_spde_bias_table(cfg)

if nargin<1 || isempty(cfg), cfg=struct(); end
here=fileparts(mfilename('fullpath'));
defaults=struct('theta',0.8,'alpha',1.9,'sigma',1,'T',1, ...
    'interval',[0 1],'x0',0.5,'initialAmplitude',0,'kernelScale',1, ...
    'nSpace',2500,'R',500,'M',[12000 1800000], ...
    'mFine',3600000,'timeScheme','semiimplicit', ...
    'delta',[0.005 0.020 0.060 0.090 0.120 0.150 0.200 0.250], ...
    'seed',20261024,'batchSize',1,'maxMemoryMB',512, ...
    'kernelQuadrature',2048,'coefficientTolerance',1e-12, ...
    'digits',3,'progressSeconds',30,'checkpointEvery',10,'outputDir','');
assert(isstruct(cfg) && isscalar(cfg),'cfg must be a scalar structure.');
fields=fieldnames(cfg);
assert(all(ismember(fields,fieldnames(defaults))),'Unknown configuration field.');
names=fieldnames(defaults);
for k=1:numel(names)
    if ~isfield(cfg,names{k}), cfg.(names{k})=defaults.(names{k}); end
end
cfg.delta=cfg.delta(:)';       
cfg.M=sort(cfg.M(:)');
cfg.timeScheme=lower(char(cfg.timeScheme));
validate_configuration(cfg);

cfg.outputDir=char(cfg.outputDir);
if isempty(cfg.outputDir)
    cfg.outputDir=fullfile(here,'results_bias_0');
end

if exist(cfg.outputDir,'dir')~=7
    [ok,message]=mkdir(cfg.outputDir); assert(ok,'Cannot create output folder: %s',message);
end
[ok,attributes]=fileattrib(cfg.outputDir); assert(ok,'Cannot resolve output folder.');
cfg.outputDir=attributes.Name;
checkpointFile=fullfile(cfg.outputDir,'bias_checkpoint.mat');


n=cfg.nSpace; J=n-1; ell=diff(cfg.interval); h=ell/n;
j=(1:J)';
lambdaH=(4/h^2)*sin(j*pi/(2*n)).^2;
muH=lambdaH.^(cfg.alpha/2);
muContinuous=(j*pi/ell).^cfg.alpha; 
initial=zeros(J,1);

initial(1)=cfg.initialAmplitude*sqrt(ell/2);
[rho,noiseSD,timeInfo]=euler_observation_transition(cfg,muH);


[z,w]=gauss_legendre(cfg.kernelQuadrature);
normalizer=sqrt(sum(w.*bump(z).^2))/cfg.kernelScale;
assert(isfinite(normalizer) && normalizer>0,'Invalid kernel normalizer.');
kernelNormSquared=sum(w.*abs(bump(z)/normalizer).^2);
kernelNorm=sqrt(kernelNormSquared);
kappaFull=kernel_projections(cfg,J,z,w,normalizer);


[z2,w2]=gauss_legendre(2*cfg.kernelQuadrature);
normalizer2=sqrt(sum(w2.*bump(z2).^2))/cfg.kernelScale;
kappaCheck=kernel_projections(cfg,J,z2,w2,normalizer2);
quadratureGap=sqrt(sum((kappaFull-kappaCheck).^2,1))./ ...
    max(sqrt(sum(kappaCheck.^2,1)),realmin);
assert(max(quadratureGap)<1e-8,'Increase kernelQuadrature: kernel integrals are unresolved.');


kappa=kappaFull;
cutoff=cfg.coefficientTolerance*max(abs(kappaFull),[],1);
kappa(bsxfun(@le,abs(kappa),cutoff))=0;
fullEnergy=sum(bsxfun(@times,muH,kappaFull.^2),1);
skippedEnergy=sum(bsxfun(@times,muH,(kappaFull-kappa).^2),1);
omittedFraction=skippedEnergy./max(fullEnergy,realmin);
assert(max(omittedFraction)<1e-12,'Reduce coefficientTolerance.');
kernelMassGap=abs(kernelNormSquared-sum(kappa.^2,1))/kernelNormSquared;
continuousEnergy=sum(bsxfun(@times,muContinuous,kappaFull.^2),1);
fdEnergyGap=abs(fullEnergy-continuousEnergy)./max(continuousEnergy,realmin);

fprintf('Fresh Bias simulation: R=%d; M1=%d; M2=%d.\n',cfg.R,cfg.M);
fprintf('theta=%.6g, alpha=%.6g, sigma=%.6g, T=%.6g, x0=%.6g, A=%.6g.\n', ...
    cfg.theta,cfg.alpha,cfg.sigma,cfg.T,cfg.x0,cfg.initialAmplitude);
fprintf('n=%d, J=%d, h=%.8g; scheme=%s, mFine=%d, dtFine=%.8g.\n', ...
    n,J,h,cfg.timeScheme,cfg.mFine,timeInfo.dtFine);
fprintf('delta:'); fprintf(' %.3f',cfg.delta); fprintf('\n');
fprintf('Kernel L2 norm=%.8g; active observed modes=%d/%d.\n', ...
    kernelNorm,sum(any(kappa~=0,2)),J);
fprintf('Max kernel mass gap=%.3g; max FD/continuous energy gap=%.3g.\n', ...
    max(kernelMassGap),max(fdEnergyGap));
fprintf('Spatial diagnostics are recorded for these finite n and delta values.\n');
fprintf('Output folder:\n%s\n',cfg.outputDir);


thetaHat=simulate_estimates(cfg,muH,kappa,initial,rho,noiseSD,checkpointFile);


D=numel(cfg.delta);
bias=zeros(2,D); empiricalSD=zeros(2,D); biasMCSE=zeros(2,D);
rows=zeros(2*D,18);
for mm=1:2
    for dd=1:D
        sample=thetaHat(:,dd,mm);
        err=sample-cfg.theta;
        bias(mm,dd)=mean(err);              
        empiricalSD(mm,dd)=std(sample,0);   
        biasMCSE(mm,dd)=empiricalSD(mm,dd)/sqrt(cfg.R);
        rows((mm-1)*D+dd,:)=[cfg.delta(dd) cfg.M(mm) cfg.R cfg.theta ...
            bias(mm,dd) mean(sample) empiricalSD(mm,dd) biasMCSE(mm,dd) ...
            cfg.alpha cfg.x0 cfg.initialAmplitude cfg.sigma cfg.T ...
            n cfg.mFine kernelNorm kernelMassGap(dd) fdEnergyGap(dd)];
    end
end
statistics=array2table(rows,'VariableNames', ...
    {'Delta','M','R','ThetaTrue','Bias','MeanEstimate','EmpiricalSD','BiasMCSE', ...
     'Alpha','X0','InitialAmplitude','Sigma','T','N','mFine','KernelNorm', ...
     'KernelMassGap','FDEnergyRelativeGap'});

files=struct('latex',fullfile(cfg.outputDir,'bias_table.tex'), ...
    'csv',fullfile(cfg.outputDir,'bias_statistics.csv'), ...
    'mat',fullfile(cfg.outputDir,'bias_results.mat'));
results=struct('config',cfg,'thetaHat',thetaHat,'Bias',bias, ...
    'EmpiricalSD',empiricalSD,'BiasMCSE',biasMCSE,'statistics',statistics, ...
    'nModes',J,'h',h,'lambdaH',lambdaH,'muH',muH,'kappa',kappa, ...
    'initial',initial,'kernelNorm',kernelNorm,'kernelNormSquared',kernelNormSquared, ...
    'timeInfo',timeInfo,'quadratureGap',quadratureGap, ...
    'kernelMassGap',kernelMassGap,'fdEnergyGap',fdEnergyGap, ...
    'omittedEnergyFraction',omittedFraction,'files',files);


save(files.mat,'results','-v7');
writetable(statistics,files.csv);
write_bias_latex(files.latex,cfg,bias);
if exist(checkpointFile,'file')==2, delete(checkpointFile); end
disp(statistics(:,{'Delta','M','R','Bias','BiasMCSE'}));
fprintf('Completed. One LaTeX table saved to:\n%s\n',files.latex);
end

function estimates=simulate_estimates(cfg,mu,kappa,initial,rho,noiseSD,checkpointFile)

Mstore=cfg.M(2); D=numel(cfg.delta);

bytesPerPath=8*(Mstore+1)*(2*D+10);
memoryBatch=floor(cfg.maxMemoryMB*1024^2/bytesPerPath);
assert(memoryBatch>=1,'Increase maxMemoryMB or reduce M2/delta count.');
Bmax=min([cfg.batchSize cfg.R memoryBatch]);
active=find(any(kappa~=0,2));
assert(~isempty(active),'The measurement kernel has no active modes.');
estimates=NaN(cfg.R,D,2);
savedRng=rng;
guard=onCleanup(@() rng(savedRng)); 
rng(cfg.seed,'twister');
lastCheckpoint=0;
progressTimer=tic;
for first=1:Bmax:cfg.R
    last=min(first+Bmax-1,cfg.R); B=last-first+1;
    Y=zeros(Mstore+1,B,D);
    Z=zeros(Mstore+1,B,D);
    for kk=1:numel(active)
        k=active(kk);
        
        xi=randn(Mstore,B);
        u=filter(noiseSD(k),[1 -rho(k)],xi, ...
            rho(k)*initial(k)*ones(1,B),1);
        clear xi;
        targets=find(kappa(k,:)~=0);
        for dd=targets(:)'
            weight=kappa(k,dd);
            Y(1,:,dd)=Y(1,:,dd)+weight*initial(k);
            Z(1,:,dd)=Z(1,:,dd)-mu(k)*weight*initial(k);
            Y(2:end,:,dd)=Y(2:end,:,dd)+weight*u;
            Z(2:end,:,dd)=Z(2:end,:,dd)-mu(k)*weight*u;
        end
        clear u;
        if toc(progressTimer)>=cfg.progressSeconds
            fprintf('Paths %d:%d; observed modes %d/%d.\n',first,last,kk,numel(active));
            progressTimer=tic;
        end
    end
    for mm=1:2
        stride=Mstore/cfg.M(mm);
        index=1:stride:Mstore+1;
        dtObs=cfg.T/cfg.M(mm);
        for dd=1:D
            Yo=Y(index,:,dd); Zo=Z(index,:,dd);
            
            numerator=sum(Zo(1:end-1,:).*diff(Yo,1,1),1);
            denominator=dtObs*sum(Zo(1:end-1,:).^2,1);
            assert(all(isfinite(denominator) & denominator>0),'Invalid estimator denominator.');
            estimate=numerator./denominator;
            assert(all(isfinite(estimate)),'Nonfinite estimator.');
            estimates(first:last,dd,mm)=estimate';
        end
    end
    clear Y Z Yo Zo;
    fprintf('Completed %d/%d Monte Carlo paths.\n',last,cfg.R);
    progressTimer=tic;
    if last-lastCheckpoint>=cfg.checkpointEvery || last==cfg.R
        
        completedPaths=last;
        thetaHat=estimates; 
        randomState=rng; 
        save(checkpointFile,'thetaHat','completedPaths','cfg','randomState','-v7');
        clear thetaHat;
        lastCheckpoint=last;
    end
end
end

function kappa=kernel_projections(cfg,J,z,w,normalizer)

ell=diff(cfg.interval);
frequency=(1:J)'*pi/ell;
phase=sin(frequency*(cfg.x0-cfg.interval(1)));
if abs(cfg.x0-mean(cfg.interval))<10*eps(max(1,ell))
    phase(2:2:end)=0; 
end
weighted=w.*bump(z)/normalizer;
kappa=zeros(J,numel(cfg.delta));
for dd=1:numel(cfg.delta)
    d=cfg.delta(dd);
    kappa(:,dd)=sqrt(2*d/ell)*phase.*(cos((frequency*d)*z')*weighted);
end
end

function y=bump(z)

y=zeros(size(z));
inside=abs(z)<1;
y(inside)=exp(-10./(1-z(inside).^2));
end

function [z,w]=gauss_legendre(N)

half=ceil(N/2);
k=(1:half)';
x=cos(pi*(k-0.25)/(N+0.5));
for iter=1:30
    [PN,PNminus]=legendre_pair(N,x);
    derivative=N*(x.*PN-PNminus)./(x.^2-1);
    correction=PN./derivative;
    x=x-correction;
    if max(abs(correction))<5e-15, break; end
end
assert(iter<30,'Gauss-Legendre Newton iteration failed.');
[PN,PNminus]=legendre_pair(N,x);
derivative=N*(x.*PN-PNminus)./(x.^2-1);
weight=2./((1-x.^2).*derivative.^2);
if mod(N,2)==0
    z=[-x;flipud(x)]; w=[weight;flipud(weight)];
else
    x(end)=0;
    z=[-x;flipud(x(1:end-1))]; w=[weight;flipud(weight(1:end-1))];
end
end

function [PN,PNminus]=legendre_pair(N,x)
P0=ones(size(x)); P1=x;
for k=2:N
    P2=((2*k-1)*x.*P1-(k-1)*P0)/k;
    P0=P1; P1=P2;
end
PN=P1; PNminus=P0;
end

function [rhoObs,sdObs,info]=euler_observation_transition(cfg,mu)

dt=cfg.T/cfg.mFine;
rate=cfg.theta*mu;
stabilityNumber=max(rate)*dt;
switch cfg.timeScheme
    case 'semiimplicit'
        r=1./(1+rate*dt);
        b=cfg.sigma*sqrt(dt)*r;
    case 'explicit'
        r=1-rate*dt;
        b=cfg.sigma*sqrt(dt)*ones(size(rate));
        if any(abs(r)>=1)
            error('SPDE:ExplicitUnstable', ...
                ['Explicit EM is unstable: theta*muMax*dt=%.6g >=2. ' ...
                 'Use semiimplicit or a sufficiently large mFine divisible by both M.'], ...
                stabilityNumber);
        end
end
s=cfg.mFine/cfg.M(2);
rhoObs=r.^s;
logR2=2*log(abs(r));
geometric=expm1(s*logR2)./expm1(logR2);
geometric(r.^2==1)=s;
sdObs=b.*sqrt(geometric);
assert(all(isfinite(sdObs) & sdObs>0),'Invalid Euler innovation variance.');
info=struct('dtFine',dt,'fineSteps',cfg.mFine, ...
    'stepsPerStoredObservation',s,'stabilityNumber',stabilityNumber, ...
    'scheme',cfg.timeScheme,'fineRho',r,'fineNoiseSD',b);
end

function write_bias_latex(path,cfg,bias)

fid=fopen(path,'w'); assert(fid>=0,'Cannot write LaTeX table: %s',path);
guard=onCleanup(@() fclose(fid)); 
put=@(s)fprintf(fid,'%s\n',s);
put('% Requires \usepackage{booktabs} in the document preamble.');
put(sprintf('%% R=%d; theta=%.12g; alpha=%.12g; sigma=%.12g; T=%.12g.', ...
    cfg.R,cfg.theta,cfg.alpha,cfg.sigma,cfg.T));
put(sprintf('%% x0=%.12g; A=%.12g; n=%d; M1=%d; M2=%d; mFine=%d.', ...
    cfg.x0,cfg.initialAmplitude,cfg.nSpace,cfg.M,cfg.mFine));
put('% Values are signed mean estimation errors; display rounding only.');
put('\begin{table}[htbp]');
put('    \centering');
put('    \caption{Bias of the estimators.}');
put('    \label{tab:bias}');
put('    \renewcommand{\arraystretch}{1}');
put('    \setlength{\tabcolsep}{3pt}');
put(['    \begin{tabular*}{\textwidth}{l @{\extracolsep{\fill}} *{' ...
    num2str(numel(cfg.delta)) '}{c}}']);
put('        \toprule');
headers=arrayfun(@(d)sprintf('%.3f',d),cfg.delta,'UniformOutput',false);
put(['        $\delta$ & ' strjoin(headers,' & ') ' \\']);
put('        \midrule');
for mm=1:2
    values=cell(1,numel(cfg.delta));
    for dd=1:numel(cfg.delta)
        values{dd}=sprintf('%.*f',cfg.digits,bias(mm,dd));
        
        if str2double(values{dd})==0
            values{dd}=sprintf('%.*f',cfg.digits,0);
        end
    end
    name=['$\mathrm{Bias}(\widehat{\vartheta}_{M_' num2str(mm) ',\delta})$'];
    put(['        ' name ' & ' strjoin(values,' & ') ' \\']);
end
put('        \bottomrule');
put('    \end{tabular*}');
put('\end{table}');
end

function validate_configuration(cfg)
scalarFields={'theta','alpha','sigma','T','x0','initialAmplitude','kernelScale'};
for k=1:numel(scalarFields)
    validateattributes(cfg.(scalarFields{k}),{'numeric'},{'real','finite','scalar'});
end
assert(cfg.theta>0 && cfg.sigma>0 && cfg.T>0 && cfg.kernelScale>0, ...
    'theta, sigma, T and kernelScale must be positive.');
assert(cfg.alpha>1 && cfg.alpha<=2,'Require 1<alpha<=2.');
validateattributes(cfg.interval,{'numeric'},{'real','finite','vector','numel',2});
assert(diff(cfg.interval)>0,'Invalid spatial interval.');
validateattributes(cfg.nSpace,{'numeric'},{'real','finite','scalar','integer','>=',4});
validateattributes(cfg.R,{'numeric'},{'real','finite','scalar','integer','>=',2});
validateattributes(cfg.M,{'numeric'},{'real','finite','integer','>=',2,'numel',2});
assert(cfg.M(1)<cfg.M(2) && mod(cfg.M(2),cfg.M(1))==0, ...
    'Require M1<M2 and M2 divisible by M1.');
validateattributes(cfg.mFine,{'numeric'},{'real','finite','scalar','integer','positive'});
assert(cfg.mFine>=cfg.M(2) && all(mod(cfg.mFine,cfg.M)==0), ...
    'mFine must be divisible by both M1 and M2.');
assert(ismember(cfg.timeScheme,{'semiimplicit','explicit'}),'Unknown timeScheme.');
validateattributes(cfg.delta,{'numeric'},{'real','finite','positive','vector','nonempty'});
assert(numel(unique(cfg.delta))==numel(cfg.delta),'delta values must be distinct.');
assert(cfg.x0-max(cfg.delta)>cfg.interval(1) && ...
    cfg.x0+max(cfg.delta)<cfg.interval(2), ...
    'Every kernel support must stay strictly inside the interval.');
validateattributes(cfg.seed,{'numeric'}, ...
    {'real','finite','scalar','integer','>=',0,'<=',2^32-1});
validateattributes(cfg.batchSize,{'numeric'},{'real','finite','scalar','integer','positive'});
validateattributes(cfg.maxMemoryMB,{'numeric'},{'real','finite','scalar','positive'});
validateattributes(cfg.kernelQuadrature,{'numeric'}, ...
    {'real','finite','scalar','integer','>=',32});
validateattributes(cfg.coefficientTolerance,{'numeric'}, ...
    {'real','finite','scalar','>=',0,'<',1});
validateattributes(cfg.digits,{'numeric'},{'real','finite','scalar','integer','>=',0,'<=',12});
validateattributes(cfg.progressSeconds,{'numeric'},{'real','finite','scalar','positive'});
validateattributes(cfg.checkpointEvery,{'numeric'}, ...
    {'real','finite','scalar','integer','positive'});
assert(ischar(cfg.outputDir) || (isa(cfg.outputDir,'string') && isscalar(cfg.outputDir)), ...
    'outputDir must be a folder path.');
end

