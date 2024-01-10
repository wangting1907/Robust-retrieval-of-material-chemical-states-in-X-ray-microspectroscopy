function [A,Scale] = LMM_SBD10(Y,bundle, A_init,S_init,lambda1,rho,maxiter_ADMM,tol_a)
   
   
%   This function minimizes the following cost with respect to A and S:
%  
%   J(A,S) = 1/2 * ||Y - BAS||_{F}^{2} + lambda1*||A||_{row,2,1} +
%   lambda2*||A||_{G,F,1} + I_{R+}(A) + I_{R+}(S)
%
%   Here, X is a spatial patch, B is a collection of endmember bundles, A is
%   the matrix of abundances for each spatial patch and S (or Scale) is a 
%   diognal matrix of scaling factors
%
%   
%
% Inputs:
%       Y :             (L-in-m*n) matrix of hyperspectral data
%       bundle:         LxQ matrix of the bundle dictionary
%       groups:         a vector that determines which endmember each atom of
%                       the bundle dictionary represents.
%       patch_idx:      an (m*n-in-1) vector that determines which spatial neighbor
%                       each pixel belongs to
%       A_init:         initial abundances
%       lambda1:        regularization parameter
%       lambda2:        regularization parameter
%       rho:            penalty parameter of the ADMM
%       maxiter_ADMM:   maximum number of iterations before the algorithm stops
%       tol_a:          stops the ADMM when relative variations (in norms) of the 
%                       abundance matrix goes below "tol_a"
%
% Output:
%       A:              (QxN)matrix of abundance matrix
%       Scale:          the diogonal matrix (NxN) of scaling factors 
%
% Author: Saeideh Azar
% Last edit: 2021-6-5
% 
%%
P=size(bundle,2);
N = size(Y,2);      % total number of pixels
B = bundle;
BtBrhoI = inv(B'*B +rho*eye(P));
BB=B'*Y;
%% ADMM 
% initialization
A1 = A_init ./sum(A_init,1);
S=sparse(1:N,1:N,S_init,N,N);
C = zeros(size(A1));
U= A1;
W = A1;
M = A1*S;
D = C;
F = C;
T=(spdiags(S));
G=zeros(N,1);
A1A=((S*S')+2*speye(N))';
LL = ichol(A1A);
net=denoisingNetwork('DnCNN');
z_3d_dn = zeros([sqrt(N), sqrt(N),P]);
for i = 1:maxiter_ADMM  % main loop of ADMM
        
        A_old = A1;
        
        %update A
        A1A=((S*S')+2*speye(N))';
        bb=((M*S'-C*S'+U-D+W-F))';
        parfor l=1:P
           A11(:,l)=cgs(A1A,bb(:,l),1e-4,10,LL,[],(A1(l,:))');
        end 
        A1 = A11'./sum(A11',1);
        %updata S
        b=(sum(A1.*(M-C),1)'+T-G);     %T,G vector
        AtA=sparse(1:N,1:N,sum(A1.*A1,1)',N,N)+speye(N);
        S=AtA\b;
       
        u_tilde = A1+D; 
        % Denoising in 3D image domain
       z_3d = permute(reshape(u_tilde,[P, sqrt(N), sqrt(N)]), [2,3,1]); 
  %     z_3d_dn = zeros(size(z_3d)); % Initialize denoised 3D image
%%     denoiser: bandwise
        parfor j = 1:P
           [z_3d_dn(:, :, j)]= Denoiser(squeeze(z_3d(:, :, j)),net); %DnCNN
        end
        U = reshape(permute(z_3d_dn, [3, 1, 2]), [P, N]);
    

        %update W
        W = max(A1+F,0);
        
        %update T
        T = max(S+G,0);
        
        
        % dual updates:
        D = D + A1 - U;
        F = F + A1 - W;
        G = G + S - T;
      
        S=sparse(1:N,1:N,S,N,N);
        %update M
        M = BtBrhoI*(BB+rho*(A1*S)+rho*C);           %improve time
        C = C + A1*S- M;
        rel_A = norm(A_old-A1,'fro')/norm(A_old,'fro')^2;
        
        if i>1 && rel_A < tol_a
            break
        end
        
end    % end of ADMM
    
A= A1;  
Scale=spdiags(S);   %支持CPU格式
%Scale=diag(S);       %支持GPU格式
end        % end of each patch



function [scaled_output_from_denoiser] = Denoiser(utilde,net)
% u-update using a call to DnCNN
ut=utilde(:);
lb = min(ut);
ub = max(ut);
utilde_scaled = (utilde - lb) / (ub - lb);
output_from_denoiser = denoiseImage(utilde_scaled,net);
% Rescale     
scaled_output_from_denoiser = lb + output_from_denoiser * (ub - lb);
end 