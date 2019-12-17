%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_train_small;
load mnist_test;

tic;
%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.00625; %1  %...
% Weight regularization parameter
hyperparameters.weight_regularization = 0;   %...
% Number of iterations
hyperparameters.num_iterations = 500;  %...
% Logistics regression weights
% TODO: Set random weights.
weights = randn((size(train_inputs,2)+1),1);
weights_small = weights;

cross_entropy_train = zeros( hyperparameters.num_iterations, 1 );
cross_entropy_train_small = cross_entropy_train;
cross_entropy_valid = cross_entropy_train;
cross_entropy_valid_small = cross_entropy_train;

%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

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
                                
    [cross_entropy_train(t), frac_correct_train] = evaluate(train_targets, predictions);
    [cross_entropy_train_small(t), frac_correct_train_samll] = evaluate(train_targets_small, predictions_small);
   
    %????
	% Find the fraction of correctly classified validation examples.
	%[temp, temp2, frac_correct_valid] = logistic(weights, ...
    %                                             valid_inputs, ...
    %                                             valid_targets, ...
    %                                             hyperparameters);

    if isnan(f) || isinf(f)
		error('nan/inf error');
    end
    
    
    if isnan(f_small) || isinf(f_small)
		error('nan/inf error');
	end

    %% Update parameters.
    weights = weights - hyperparameters.learning_rate .* df;%/ N;
    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid(t), frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
    weights_small = weights_small - hyperparameters.learning_rate .* df_small;%/N_small;
    predictions_valid_small = logistic_predict(weights_small, valid_inputs);
    [cross_entropy_valid_small(t), frac_correct_valid_small] = evaluate(valid_targets, predictions_valid_small);
    
    predictions_test = logistic_predict(weights, test_inputs);
    [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
    
	%% Print some stats.
	fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f\n TRAIN CE %.6f TRAIN FRAC:%2.2f\t VALIC_CE %.6f VALID FRAC:%2.2f\t ',...
			t, f/N, cross_entropy_train(t), frac_correct_train*100, cross_entropy_valid(t), frac_correct_valid*100);
    fprintf(1, 'TEST CE %.6f TEST FRAC:%2.2f\n',cross_entropy_test, frac_correct_test*100);

end
toc;

figure;
subplot(1,2,1);
hold on;
title('mnist\_train', 'FontSize', 15);
plot(1:hyperparameters.num_iterations, cross_entropy_train, 'LineWidth', 1.5);
plot(1:hyperparameters.num_iterations, cross_entropy_valid, 'LineWidth', 1.5);
l = legend('train', 'valid');
set(l, 'FontSize', 15);
xlabel('num of iteration', 'FontSize', 15);
ylabel('cross\_entropy', 'FontSize', 15);

subplot(1,2,2);
hold on;
title('mnist\_train\_small', 'FontSize', 15);
plot(1:hyperparameters.num_iterations, cross_entropy_train_small, 'LineWidth', 1.5);
plot(1:hyperparameters.num_iterations, cross_entropy_valid_small, 'LineWidth', 1.5);
l = legend('train\_small', 'valid');
set(l, 'FontSize', 15);
xlabel('num of iteration', 'FontSize', 15);
ylabel('cross\_entropy', 'FontSize', 15);
