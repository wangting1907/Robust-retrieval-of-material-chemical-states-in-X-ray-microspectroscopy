function [u,time] = Denoiser(utilde)
    
% Set path to DnCNN
% addpath 'DnCNN/utilities';
% addpath 'DnCNN/model';
% addpath 'DnCNN/model/specifics';
 disp(['=== load the pre-trained network...']);
        net = denoisingNetwork('DnCNN');
t1=clock;
%n2=size(utilde,2);
% u-update using a call to DnCNN
ut=utilde(:);
lb = min(ut);
ub = max(ut);
% Scale utilde to [0 1]
utilde_scaled = (utilde - lb) / (ub - lb);
output_from_denoiser = denoiseImage(utilde_scaled,net);
% Rescale     
scaled_output_from_denoiser = lb + output_from_denoiser * (ub - lb);

u=scaled_output_from_denoiser;
    
% if ~DISPLAY
%     toc(t_start);
% end
t2=clock;
time = etime(t2,t1);
end 