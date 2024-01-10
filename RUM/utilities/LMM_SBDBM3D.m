function [A,Scale] = LMM_SBDBM3D(Y,bundle, A_init,S_init,lambda1,rho,maxiter_ADMM,tol_a)
   
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
S=sparse(1:N,1:N,S_init,N,N);
C = zeros(size(A1));
U= A1;
%U2= A1*Dv';
%V = A1;
W = A1;

M = A1*S;
D = C;
%D2=C;
F = C;
T=(spdiags(S));
G=zeros(N,1);
%lap=Dh'*Dh+Dv'*Dv;
A1A=((S*S')+2*speye(N))';
LL = ichol(A1A);
for i = 1:maxiter_ADMM  % main loop of ADMM
        
        A_old = A1;
        
        %update A
        A1A=((S*S')+2*speye(N))';
        bb=((M*S'-C*S'+U-D+W-F))';
        parfor l=1:P
           A11(:,l)=cgs(A1A,bb(:,l),1e-4,10,LL,[],(A1(l,:))');
        end 
        A1 = A11'./sum(A11',1);
        %A1 = A11';
        %updata S
        b=(sum(A1.*(M-C),1)'+T-G);     %T,G vector
        %AtA=sum(A1.*A1,1)'+1;
        AtA=sparse(1:N,1:N,sum(A1.*A1,1)',N,N);
        AtA=(AtA+speye(N));
        S=AtA\b;
       
        u_tilde = A1+D; 
        % Denoising in 3D image domain
        z_3d = permute(reshape(u_tilde,[P, sqrt(N), sqrt(N)]), [2,3,1]); 
%%     denoiser: bandwise
        sigma = sqrt(lambda1/ rho);
        z_3d_dn = zeros(size(z_3d)); % Initialize denoised 3D image
        parfor j = 1:P
           [~, z_3d_dn(:, :, j)]= BM3D(1, z_3d(:, :, j), sigma*255, 'lc'); % BM3D    
        end
        U = reshape(permute(z_3d_dn, [3, 1, 2]), [P, N]);
        %update V
        %V = prox_GF1(A1+E,groups,lambda2/rho);

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
        M = BtBrhoI*(BB+rho*(A1*S)+rho*C); %improve time
        C = C + A1*S- M;
        rel_A = norm(A_old-A1,'fro')/norm(A_old,'fro')^2;
        
        if i>1 && rel_A < tol_a
            break
        end
        
end    % end of ADMM
    
A= A1;  
Scale=spdiags(S);

end        % en