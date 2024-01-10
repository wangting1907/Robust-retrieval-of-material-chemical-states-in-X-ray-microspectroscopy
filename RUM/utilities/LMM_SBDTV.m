function [A,Scale] = LMM_SBD7(Y,bundle, A_init,S_init,lambda1,rho,maxiter_ADMM,tol_a)
   
   
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

%P = length(groups); %  (Q in paper): total number of endmember signature or dictionry size 
P=size(bundle,2);
N = size(Y,2);      % total number of pixels

%A_init = A_init ./sum(A_init,1);  % imposing sum-to-one constraint

B = bundle;
BtBrhoI = B'*B +rho*eye(P);
BtBrhoI=inv(BtBrhoI);
BB=B'*Y;
%% ADMM 
% initialization
A1 = A_init ./sum(A_init,1);
%S = diag(ones(N_patch,1));
n=sqrt(N);
m=n;
Dh=spdiags([-ones(n,1) ones(n,1)],[0 1],n,n);   Dh(n,:) = 0;    Dh = kron(Dh,speye(m));
Dv=spdiags([-ones(m,1) ones(m,1)],[0 1],m,m);   Dv(m,:) = 0;    Dv = kron(speye(n),Dv);
%S = diag(S_init(patch_idx==k));
S=sparse(1:N,1:N,S_init,N,N);
C = zeros(size(A1));
U1= A1*Dh';
U2= A1*Dv';
%V = A1;
W = A1;

M = A1*S;
D1 = C;
D2=C;
F = C;
T=(spdiags(S));
G=zeros(N,1);
lap=Dh'*Dh+Dv'*Dv;
A1A=((S*S')+lap+speye(N))';
LL = ichol(A1A);
%S=gpuArray(S);
for i = 1:maxiter_ADMM  % main loop of ADMM
        
        A_old = A1;
        
        %update A
        A1A=((S*S')+lap+speye(N))';
        %A1A=((S*S')+lap+eye(N))';
        bb=((M*S'-C*S'+(U1-D1)*Dh+(U2-D2)*Dv+W-F))';
        for l=1:P
           A11(:,l)=cgs(A1A,bb(:,l),1e-4,10,LL,[],(A1(l,:))');
        end 
        A1 = A11'./sum(A11',1);
        %updata S
        b=(sum(A1.*(M-C),1)'+T-G);     %T,G vector
        %AtA=sum(A1.*A1,1)'+1;
        AtA=sparse(1:N,1:N,sum(A1.*A1,1)',N,N);
        AtA=(AtA+speye(N));
        S=AtA\b;
       
        %S=cgs(AtA,b,1e-4,5,[],[],spdiags(S));
      
       
        %update U
        %U = prox_row21(A1+D,lambda1/rho);
        parfor j=1:P
        U1(j,:) = shrink(A1(j,:)*Dh'+D1(j,:),lambda1/rho); 
        U2(j,:) = shrink(A1(j,:)*Dv'+D2(j,:),lambda1/rho); 
        end
        %update V
        %V = prox_GF1(A1+E,groups,lambda2/rho);

        %update W
        W = max(A1+F,0);
        
        %update T
        T = max(S+G,0);
        
        
        % dual updates:
        D1 = D1 + A1*Dh' - U1;
        D2 = D2 + A1*Dv' - U2;
        F = F + A1 - W;
        G = G + S - T;
      
        S=sparse(1:N,1:N,S,N,N);
        %update M
        M = BtBrhoI*(BB+rho*(A1*S)+rho*C); %improve time
        C = C + A1*S- M;
        rel_A = norm(A_old-A1,'fro')/norm(A_old,'fro')^2;
        
        if i>1 && rel_A < tol_a
            break
        end
        
end    % end of ADMM
    
A= A1;  
%Scale=spdiags(S);   %支持cpu 格式；
Scale=diag(S);       %支持Gpu格式；

end        % end of each patch
                    


function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end