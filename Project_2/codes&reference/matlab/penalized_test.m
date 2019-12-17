%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_test;
load mnist_train_small;

tic;   

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.00625;   %...
% Weight regularization parameter
hyperparameters.weight_regularization = 0.1;   %...
% Number of iterations
hyperparameters.num_iterations = 300;  %...
% Logistics regression weights
% TODO: Set random weights.

frac_test = 0;
cross_test = 0;
frac_test_small = 0;
cross_test_small = 0;

for k = 1:10
    weights = randn((size(train_inputs,2)+1),1);
    weights_small = weights;
    N = size(train_inputs,1);
    N_small = size(train_inputs_small,1);
    
    for t = 1:hyperparameters.num_iterations
        
        % Find the negative log likelihood and derivative w.r.t. weights.
        [f, df, predictions] = logistic_pen(weights, ...
            train_inputs, ...
            train_targets, ...
            hyperparameters);
        [f_small, df_small, predictions_small] = logistic_pen(weights_small, ...
            train_inputs_small, ...
            train_targets_small, ...
            hyperparameters);
        
        [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);
        [cross_entropy_train_small, frac_correct_train_small] = evaluate(train_targets_small, predictions_small);
                                                          
        if isnan(f) || isinf(f)
            error('nan/inf error');
        end
        if isnan(f_small) || isinf(f_small)
            error('nan/inf error');
        end
        
        %% Update parameters.
        weights = weights - hyperparameters.learning_rate .* df;%/N;
        predictions_test = logistic_predict(weights, test_inputs);
        [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
        
        weights_small = weights_small - hyperparameters.learning_rate .* df_small;%/N_small;
        predictions_test_small = logistic_predict(weights_small, test_inputs);
        [cross_entropy_test_small, frac_correct_test_small] = evaluate(test_targets, predictions_test_small);
        
        %% Print some stats.
        fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
            t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_test, frac_correct_test*100);
        
    end
    frac_test = frac_test + frac_correct_test;
    cross_test = cross_test + cross_entropy_test;
    frac_test_small = frac_test_small  + frac_correct_test_small;
    cross_test_small = cross_test_small + cross_entropy_test_small;
end

toc;

frac_test = frac_test/10;
cross_test = cross_test/10;
frac_test_small = frac_test_small/10;
cross_test_small = cross_test_small/10;

fprintf('TEST CE:  %.6f TEST FRAC: %2.2f\n TEST SMALL CE: %.6f TEST SMALL FRAC: %2.2f\n',...
             cross_test, frac_test*100, cross_test_small, frac_test_small*100);
    


