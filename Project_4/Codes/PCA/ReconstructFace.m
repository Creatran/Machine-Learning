load freyface.mat
X = double(X);

[m, N] = size(X);

Xctr = X - repmat(mean(X, 2), 1, N); 
[Vctr, Dctr] = eig(Xctr*Xctr' /N);
[lambda_ctr, order] = sort(diag(Dctr), 'descend'); 
Vctr = Vctr(:, order);
V = Vctr(:, 1:2);
Yctr = V' * X;

y1 = randi([1,255], 560, 1);
figure;
showfreyface(y1);
figure;
showfreyface(reconstruct(V'*(y1-mean(X,2)),V)+mean(X,2));

% Adding noise (choosing the 100 th Fray's face)
X_noise = X(:, 50) + 10*randn(m,1);
figure;
showfreyface(X_noise);
figure;
showfreyface(reconstruct(V'*(X_noise-mean(X,2)),V)+mean(X,2));
