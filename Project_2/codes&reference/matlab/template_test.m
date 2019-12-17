%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_train_small;
load mnist_test;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.00375; %0.6  %...
% Weight regularization parameter
hyperparameters.weight_regularization = 0;   %...
% Number of iterations
hyperparameters.num_iterations = 500;  %...
% Logistics regression weights
% TODO: Set random weights.
weights = randn((size(train_inputs,2)+1),1);
weights_small = weights;

N = size(train_inputs,1);
N_small = size(train_inputs_small,1);

%% Begin learning with gradient descent.
for t = 1:hyperparameters.num_iterations

	%% TODO: You will need to modify this loop to create plots etc.

	% Find the negative log likelihood and derivative w.r.t. weights.
	[f, df, predictions] = logistic(weights, ...
                                    train_inputs, ...
                                    train_targets, ...
                                    hyperparameters);
 
    [f_small, df_small, predictions_small] = logistic(weights_small, ...
                                                      train_inputs_small, ...
                                                      train_targets_small, ...
                                                      hyperparameters);
                                
    [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);
    [cross_entropy_train_small, frac_correct_train_samll] = evaluate(train_targets_small, predictions_small);

    if isnan(f) || isinf(f)
		error('nan/inf error');
    end
    
    
    if isnan(f_small) || isinf(f_small)
		error('nan/inf error');
	end

    %% Update parameters.
    weights = weights - hyperparameters.learning_rate .* df;%/ N;
    predictions_test = logistic_predict(weights, test_inputs);
    [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
    
    weights_small = weights_small - hyperparameters.learning_rate .* df_small;%/N_small;
    predictions_test_small = logistic_predict(weights_small, test_inputs);
    [cross_entropy_test_small, frac_correct_test_small] = evaluate(test_targets, predictions_test_small);
    
	%% Print some stats.
	fprintf('ITERATION:%4i TEST CE %.6f TEST FRAC:%2.2f\t TEST SMALL CE %.6f TEST SMALL FRAC:%2.2f\n',...
             t, cross_entropy_test, frac_correct_test*100, cross_entropy_test_small, frac_correct_test_small*100);
    

end
