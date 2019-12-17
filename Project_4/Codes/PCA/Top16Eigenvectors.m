load freyface.mat
X = double(X);
[m, N] = size(X);

[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun), 'descend'); 
Vun = Vun(:, order);


Xctr = X - repmat(mean(X, 2), 1, N); 
[Vctr, Dctr] = eig(Xctr*Xctr' /N);
[lambda_ctr, order] = sort(diag(Dctr), 'descend'); 
Vctr = Vctr(:, order);

showfreyface(Vctr(:, 1:16));
showfreyface(Vun(:, 1:16));