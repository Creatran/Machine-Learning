load freyface.mat
X = double(X);

[m, N] = size(X);
% lambda_un is the eigenvalues without removing the mean
% with a descending order.
% Vun is the eigenvectors correspond to lambda_un.
[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun), 'descend'); 
Vun = Vun(:, order);

% lambda_ctr is the eigenvalues with removing the mean
% with a descending order.
% Vctr is the eigenvectors correspond to lambda_ctr.
Xctr = X - repmat(mean(X, 2), 1, N); 
[Vctr, Dctr] = eig(Xctr*Xctr' /N);
[lambda_ctr, order] = sort(diag(Dctr), 'descend'); 
Vctr = Vctr(:, order);

figure;
plot(1:m, lambda_ctr(1:m, 1));
xlabel('number of k');
ylabel('lambda\_un');


figure;
plot(1:100, lambda_ctr(1:100, 1));
xlabel('number of k');
ylabel('lambda\_un');

% p  : the number of k
% p1 : the percentage of the first p lambdas take  in all lambdas
% S0 : the weights of the first p lambdas
% S1 : the weights of all lambdas
p = 0;      p1 = 0;
s0 = 0;     s1 = sum(lambda_ctr);
while (p1 <0.95) 
    p = p + 1; 
    s0 = s0 + lambda_ctr(p);
    p1 = s0/s1;
end
