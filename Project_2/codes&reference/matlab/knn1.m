clear all;
close all;

load mnist_train;
load mnist_valid;

[n, m] = size(valid_inputs);

r = zeros(n,1);

for k = 1:2:9
    valid_predict = run_knn(k, train_inputs, train_targets, valid_inputs);
    r(k) = sum([valid_predict==valid_targets]) / n;
end

figure;
subplot(1,2,1);
hold on;
title('Valid data', 'FontSize', 15);
plot(1:2:9,r(1:2:9));
xlabel('k', 'FontSize', 15);
ylabel('classification rate', 'FontSize', 15);



kstar = 5;
krange = [kstar-2,kstar,kstar+2];
disp(r(krange));

load mnist_test;
[n_test, m_test] = size(test_inputs);

r_test = zeros(n_test,1);

for k = 1:2:9
    test_predict = run_knn(k, train_inputs, train_targets, test_inputs);
    r_test(k) = sum([test_predict==test_targets]) / n_test;
end
%disp(r(1:2:9));

subplot(1,2,2);
hold on;
title('Test data', 'FontSize', 15);
plot(1:2:9,r_test(1:2:9));
xlabel('k', 'FontSize', 15);
ylabel('classification rate', 'FontSize', 15);