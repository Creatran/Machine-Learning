%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_test;

%%
r0=0;
ce = [];
frac=[];
N = size(train_inputs,1);
tic;
for r = 0.5:0.1:1.5
    r0 = r0+1;
    % TODO: Initialize hyperparameters.
    % Learning rate
    hyperparameters.learning_rate = r;   %...
    % Weight regularization parameter
    hyperparameters.weight_regularization = 0;   %...
    % Number of iterations
    hyperparameters.num_iterations = 300;  %...
    % Logistics regression weights
    % TODO: Set random weights.
    weights = randn((size(train_inputs,2)+1),1);
    weights_small = weights;
    
    cross_entropy_train = zeros( hyperparameters.num_iterations, 1 );
    cross_entropy_train_small = cross_entropy_train;
    cross_entropy_valid = cross_entropy_train;
    cross_entropy_valid_small = cross_entropy_train;
    
    %% Begin learning with gradient descent.
    for t = 1:hyperparameters.num_iterations
        
        %% TODO: You will need to modify this loop to create plots etc.
        
        % Find the negative log likelihood and derivative w.r.t. weights.
        [f, df, predictions] = logistic(weights, ...
            train_inputs, ...
            train_targets, ...
            hyperparameters);
        
        [cross_entropy_train(t), frac_correct_train] = evaluate(train_targets, predictions);
       
        %????
        % Find the fraction of correctly classified validation examples.
        %[temp, temp2, frac_correct_valid] = logistic(weights, ...
        %                                             valid_inputs, ...
        %                                             valid_targets, ...
        %                                             hyperparameters);
        
        if isnan(f) || isinf(f)
            error('nan/inf error');
        end
       
        %% Update parameters.
        weights = weights - hyperparameters.learning_rate .* df/ N;
        predictions_valid = logistic_predict(weights, valid_inputs);
        [cross_entropy_valid(t), frac_correct_valid] = evaluate(valid_targets, predictions_valid);
       
        predictions_test = logistic_predict(weights, test_inputs);
        [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
        
        %% Print some stats.
        fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f\n TRAIN CE %.6f TRAIN FRAC:%2.2f\t VALIC_CE %.6f VALID FRAC:%2.2f\t ',...
            t, f/N, cross_entropy_train(t), frac_correct_train*100, cross_entropy_valid(t), frac_correct_valid*100);
        fprintf(1, 'TEST CE %.6f TEST FRAC:%2.2f\n',cross_entropy_test, frac_correct_test*100);
        
    end
    ce(r0)= cross_entropy_valid(t);
    frac(r0) = frac_correct_valid*100;
end
toc;

x = 0.5:0.1:1.5;
figure;
subplot(1,2,1);
hold on;
title('cross entropy', 'FontSize', 15);
plot(x, ce, 'LineWidth', 1.5);
xlabel('rate', 'FontSize', 15);
ylabel('cross entropy', 'FontSize', 15);
subplot(1,2,2);
hold on;
plot(x, frac, 'LineWidth', 1.5);
title('fraction', 'FontSize', 15);
xlabel('rate', 'FontSize', 15);
ylabel('fraction', 'FontSize', 15);
