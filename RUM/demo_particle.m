%--------------------------------------------------------------------------
%        RUM for unmixing of the Particle dataset
%--------------------------------------------------------------------------
clear;
clc;
addpath('utilities');
addpath('BM3D');
load('images/spectrum.mat')

img= imread(['images/particle.tif']);
img=img(51:350, 90:389);   
img=im2double(img);
energy = linspace(8330,8363,1000)';
rng(10);

%%  generate data
disp('generating the ground truth chemical map....')
[m,n]=size(img);
spectrum=spectrum(:,[1,5]);
[P,L]=size(spectrum);

%% %% generate adundance 
F = 30; % number of patterns in the image
s = gen_abundance(m,n,L,F);
s(find(s<0.1))=0;
s=reshape(s,m*n,L);
parfor i =1:m*n
     tmp=s(i,:)/sum(s(i,:));
     s(i,:)=tmp;
end
M=spectrum*s';
M=M';
imgdata=max(img(:),0);
imgdata(imgdata<0.001)=0;
imgdata=10*(imgdata+0.1);

%% *scaling factor 
parfor j=1:size(spectrum,1)
    data(:,j)=M(:,j).*imgdata;
end     
%% add noise
data=reshape(data,m,n,P);
sigma_a =3;
noise = randn(size(data));
noise_std = std(noise(:));
noisydata = double(data) + sigma_a*noise/(noise_std);
%% 
X = reshape(noisydata,m*n,P);
bundle=spectrum; 
%% unmixing with FCLSU used to initialize the LMM-SBD
S_init = mean(noisydata,3);
S_init = S_init(:);
A_FCLSU=FCLSU((X./S_init)',bundle);
%% parameter setting
A_init=A_FCLSU';
rho = 400;                    % penalty parameter of the ADMM
tol_a = 10^(-6);             % stop ADMM when relative variations (in norms) of the abundance matrix goes below "tol_a" 
maxiter_ADMM = 100;          % stop ADMM after "maxiter_ADMM" iterations

% set the regularization parameters:
lambda1 = 50;

%% unmixing with the RUM for TV
tic
[A_LMMSBD3,Scale_LMMSBD3] = LMM_SBDTV(X',bundle,A_init,S_init,lambda1,rho,maxiter_ADMM,tol_a); %% LMM-ATV
time_SUM3=toc;
%% unmixing with the RUM for DnCNN
%X1= (double(X));                                %Converted to GPU format
%bundle1=(double(spectrum));
%A_init1=(double(A_FCLSU'));
tic
[A_LMMSBD10,Scale_LMMSBD10] = LMM_SBDDnCNN(X',bundle,A_init,S_init,lambda1,rho,maxiter_ADMM,tol_a);   %%LMM-DnCNN
time_SUM10=toc;

%% unmixing with the RUM for BM3D
tic
[A_LMMSBD11,Scale_LMMSBD11] = LMM_SBDBM3D(X',bundle,A_init,S_init,lambda1,rho,maxiter_ADMM,tol_a);   %%LMM-BM3D
time_SUM11=toc;

%% Edge-50 and Linear fitting
noisydata1=aux_imscale(noisydata,[0,1]);
%noisydata1=gpuArray(noisydata1);     %GPU
tic
edgepointmap = getEdgePoint(energy,noisydata1);                            %0.5 of the each pixel
sum_edge=toc;

tic
disp('unmixing with the FLCSU:')
A_FCLSU=FCLSU(X',bundle);
sum_fitting=toc;
%% 
time_traditional=[sum_edge,sum_fitting];
disp(['Running time: ' num2str(time_traditional)])

%% Evaluation--sum one 
 Cest3=A_LMMSBD3';
 Cest3(find(Cest3<0))=0;
 Cest3(find(Cest3>1))=1;
 parfor i =1:m*n
        tmp=Cest3(i,:)/sum(Cest3(i,:));
        tmp(tmp<=0.001)=0;
        Cest3(i,:)=tmp;
 end
 
 Cest10=A_LMMSBD10';
 Cest10(find(Cest10<0))=0;
 Cest10(find(Cest10>1))=1;
 parfor i =1:m*n
        tmp=Cest10(i,:)/sum(Cest10(i,:));
        tmp(tmp<=0.001)=0;
        Cest10(i,:)=tmp;
 end
 
 Cest11=A_LMMSBD11';
 Cest11(find(Cest11<0))=0;
 Cest11(find(Cest11>1))=1;
 parfor i =1:m*n
        tmp=Cest11(i,:)/sum(Cest11(i,:));
        tmp(tmp<=0.001)=0;
        Cest11(i,:)=tmp;
 end
 
 
 
%%  compute the psnr and ssim
mask=img;
mask(find(mask>0.01))=1;
mask(find(mask<=0.01))=0;
masklength=length(find(mask>0.01));
sm=s.*mask(:);
Cest3=Cest3.*mask(:);
rmse_3mm = sqrt(sum(sum(((sm-Cest3).*(sm-Cest3)).^2))/(masklength*L));
[psnr3,ssim3, correlation3]=evaluation(sm,Cest3,m,n);



Cest10=Cest10.*mask(:);
rmse_10mm = sqrt(sum(sum(((sm-Cest10).*(sm-Cest10)).^2))/(masklength*L));
rmse_img10 = sqrt(sum(sum(((imgdata-Scale_LMMSBD10).*(imgdata-Scale_LMMSBD10)).^2))/(m*n));
[psnr10,ssim10,correlation10]=evaluation(sm,Cest10,m,n);

Cest11=Cest11.*mask(:);
rmse_11mm = sqrt(sum(sum(((sm-Cest11).*(sm-Cest11)).^2))/(masklength*L));
rmse_img11 = sqrt(sum(sum(((imgdata-Scale_LMMSBD11).*(imgdata-Scale_LMMSBD11)).^2))/(m*n));
[psnr11,ssim11,correlation11]=evaluation(sm,Cest11,m,n);

time_SUM=[time_SUM3,time_SUM10,time_SUM11];
rmse_1mm=[rmse_3mm,rmse_10mm,rmse_11mm];
PSNR=[psnr3,psnr10,psnr11];
SSIM=[ssim3,ssim10,ssim11];
CORR=[correlation3,correlation10,correlation11];

%disp(['Running time: ' num2str(time_SUM)])
%disp(['FPSNR:' num2str(FPSNR)] )
disp(['rmse: ' num2str(rmse_1mm)])
disp(['psnr: ' num2str(PSNR)])
disp(['ssim: ' num2str(SSIM)])
disp(['correlation: ' num2str(CORR)])








