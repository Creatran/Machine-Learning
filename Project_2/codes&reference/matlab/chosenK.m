clear all;
close all;

load mnist_train;
load mnist_valid;

[n, m] = size(valid_inputs);
[n_train, m_train] = size(train_inputs);
t = floor(n_train/3);
step = [1, t, 2*t, n_train];
r = zeros(4,9);

for k = 1:2:9
    valid_predict = run_knn(k, train_inputs, train_targets, valid_inputs);
    r(1,k) = sum([valid_predict==valid_targets]) / n;
end

for t = 2:4
    t1_inputs = [valid_inputs; train_inputs(1:(step(t-1)-1), : ); 
        train_inputs((step(t)+1):n_train, : )];
    t1_targets = [valid_targets; train_targets(1:(step(t-1)-1), : ); 
        train_targets((step(t)+1):n_train, : )];
    t2_inputs = train_inputs(step(t-1):step(t), : ); 
    t2_targets = train_targets(step(t-1):step(t), : );
    for k = 1:2:9
        t_predict = []
        t_predict = run_knn(k, t1_inputs, t1_targets, t2_inputs);
        r(t,k) = sum([t_predict == t2_targets]) / (step(t)-step(t-1)+1);
    end
end

figure;
subplot(2,2,1);
hold on;
title('cross validation-1', 'FontSize', 15);
plot(1:2:9,r(1 , 1:2:9));
xlabel('k', 'FontSize', 15);
ylabel('classification rate', 'FontSize', 15);


subplot(2,2,2);
hold on;
title('cross validation-2', 'FontSize', 15);
plot(1:2:9,r(2 , 1:2:9));
xlabel('k', 'FontSize', 15);
ylabel('classification rate', 'FontSize', 15);

subplot(2,2,3);
hold on;
title('cross validation-3', 'FontSize', 15);
plot(1:2:9,r(3 , 1:2:9));
xlabel('k', 'FontSize', 15);
ylabel('classification rate', 'FontSize', 15);

subplot(2,2,4);
hold on;
title('cross validation-4', 'FontSize', 15);
plot(1:2:9,r(4 , 1:2:9));
xlabel('k', 'FontSize', 15);
ylabel('classification rate', 'FontSize', 15);

