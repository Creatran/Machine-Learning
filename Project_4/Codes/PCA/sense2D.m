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

Yun = Vun(:, 1:2)' * X;
Yctr = Vctr(:, 1:2)' * X;

figure;
plot(Yctr(1, :), Yctr(2, :), '.');
figure;
plot(Yun(1, :), Yun(2, :), '.');

explorefreymanifold(Yctr, X);


