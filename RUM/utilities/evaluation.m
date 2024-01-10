function [PSNR, SSIM, Correlation] = evaluation(data, original,m,n)

%%
% Subspace Modeling for Fast and High-sensitivity X-ray Chemical Imaging

%%
%SSIM=ssim(data,original);
M1=reshape(data,[m,n,size(data,2)]);
M0=reshape(original,[m,n,size(original,2)]);
%mask=img;
%mask(find(mask>0.01))=1;
k=1;
for j=1:size(M1,3)
    PSNR(k)=aux_PSNR(M1(:,:,j), M0(:,:,j));
    SSIM(k)=ssim(M1(:,:,j), M0(:,:,j));
    coe=corrcoef(M1(:,:,j), M0(:,:,j));
    Correlation(k)=coe(1,2);
    k=k+1;
end
PSNR=nanmean(PSNR);
SSIM=nanmean(SSIM);
Correlation=mean(Correlation);

% coe=corrcoef(edgepointmapOut(img>0), edgepointmapGT(img>0));
% Correlation=coe(1,2);
% Correlation=ssim(edgepointmapOut(img>0), edgepointmapGT(img>0));
end

