function C = gen_abundance(m,n,P,F)

% Simulation de P cartes d'abondances de taille
% m x n, à partir de mélange de gaussiennes

t1 = (1:m)/m;
t2 = (1:n)/n;
A = zeros(m,n,P);
for p=1:P
    L = rand(2,F);
    for k=1:F;
        A(:,:,p) = A(:,:,p) + kron(exp(-150*(t2-L(1,k)).^2),exp(-150*(t1-L(2,k)).^2)')/F;
    end
end
C = reshape(A,m*n,P);
C = C./(sum(C,2)*ones(1,P));
C = reshape(C,m,n,P);

