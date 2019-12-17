%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_train_small;

lambda = [0.001, 0.01, 0.1, 1];
cross_entropy_train_lambda = zeros(4, 1);
cross_entropy_valid_lambda = zeros(4, 1);
frac_correct_train_lambda = zeros(4, 1);
frac_correct_valid_lambda = zeros(4, 1);

cross_entropy_train_lambda_small = zeros(4, 1);
cross_entropy_valid_lambda_small = zeros(4, 1);
frac_correct_train_lambda_small = zeros(4, 1);
frac_correct_valid_lambda_small = zeros(4, 1);

tic;   
for la = 1:4
    %% TODO: Initialize hyperparameters.
    % Learning rate
    hyperparameters.learning_rate = 0.00625;   %...
    % Weight regularization parameter
    hyperparameters.weight_regularization = lambda(la);   %...
    % Number of iterations
    hyperparameters.num_iterations = 300;  %...
    % Logistics regression weights
    % TODO: Set random weights.
    
 %   cross_entropy_train = zeros( hyperparameters.num_iterations, 10 );
  %  cross_entropy_valid = cross_entropy_train;
  %  frac_correct_train = cross_entropy_train;
  %  frac_correct_valid = cross_entropy_train;
    
  %  cross_entropy_train_small = zeros( hyperparameters.num_iterations, 10 );
  %  cross_entropy_valid_small = cross_entropy_train_small;
  %  frac_correct_train_small = cross_entropy_train_small;
  %  frac_correct_valid_small = cross_entropy_valid_small;
    
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
            %cross_entropy_train
            %????
            %% Find the fraction of correctly classified validation examples.
        %    [temp, temp2, frac_correct_valid] = logistic(weights, ...
        %                                                 valid_inputs, ...
        %                                                 valid_targets, ...
        %                                                 hyperparameters);

%            [temp, temp2, frac_correct_valid_small] = logistic(weights, ...
%                                                               valid_inputs, ...
%                                                               valid_targets, ...
%                                                               hyperparameters);
            if isnan(f) || isinf(f)
                error('nan/inf error');
            end
            if isnan(f_small) || isinf(f_small)
                error('nan/inf error');
            end

            %% Update parameters.
            weights = weights - hyperparameters.learning_rate .* df;%/N;
            predictions_valid = logistic_predict(weights, valid_inputs);
            [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
            weights_small = weights_small - hyperparameters.learning_rate .* df_small;%/N_small;
            predictions_valid_small = logistic_predict(weights_small, valid_inputs);
            [cross_entropy_valid_small, frac_correct_valid_small] = evaluate(valid_targets, predictions_valid_small);
    
            %% Print some stats.
             fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
                    t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

        end
        cross_entropy_train_lambda(la) = cross_entropy_train_lambda(la) + cross_entropy_train;
        cross_entropy_valid_lambda(la) = cross_entropy_valid_lambda(la) + cross_entropy_valid;
        frac_correct_train_lambda(la) = frac_correct_train_lambda(la) + frac_correct_train;
        frac_correct_valid_lambda(la) = frac_correct_valid_lambda(la) + frac_correct_valid;
       
        cross_entropy_train_lambda_small(la) = cross_entropy_train_lambda_small(la) + cross_entropy_train_small;
        cross_entropy_valid_lambda_small(la) = cross_entropy_valid_lambda_small(la) + cross_entropy_valid_small;
        frac_correct_train_lambda_small(la) = frac_correct_train_lambda_small(la) + frac_correct_train_small;
        frac_correct_valid_lambda_small(la) = frac_correct_valid_lambda_small(la) + frac_correct_valid_small;
        
    end
    cross_entropy_train_lambda(la) = cross_entropy_train_lambda(la)/10;
    cross_entropy_valid_lambda(la) = cross_entropy_valid_lambda(la)/10;
    frac_correct_train_lambda(la) = frac_correct_train_lambda(la)/10;
    frac_correct_valid_lambda(la) = frac_correct_valid_lambda(la)/10;
    
    cross_entropy_train_lambda_small(la) = cross_entropy_train_lambda_small(la)/10;
    cross_entropy_valid_lambda_small(la) = cross_entropy_valid_lambda_small(la)/10;
    frac_correct_train_lambda_small(la) = frac_correct_train_lambda_small(la)/10;
    frac_correct_valid_lambda_small(la) = frac_correct_valid_lambda_small(la)/10;
end

toc;

figure;
subplot(2,2,1);
hold on;
title('mnist\_train', 'FontSize', 15);
plot(lambda, cross_entropy_train_lambda, 'LineWidth', 1.5);
plot(lambda, cross_entropy_valid_lambda, 'LineWidth', 1.5);
l = legend('train', 'valid');
set(l, 'FontSize', 15);
xlabel('\lambda', 'FontSize', 15);
ylabel('cross\_entropy', 'FontSize', 15);

subplot(2,2,2);
hold on;
title('mnist\_train', 'FontSize', 15);
plot(lambda, frac_correct_train_lambda, 'LineWidth', 1.5);
plot(lambda, frac_correct_valid_lambda, 'LineWidth', 1.5);
l = legend('train', 'valid');
set(l, 'FontSize', 15);
xlabel('\lambda', 'FontSize', 15);
ylabel('frac\_correct', 'FontSize', 15);


subplot(2,2,3);
hold on;
title('mnist\_train\_small', 'FontSize', 15);
plot(lambda, cross_entropy_train_lambda_small, 'LineWidth', 1.5);
plot(lambda, cross_entropy_valid_lambda_small, 'LineWidth', 1.5);
l = legend('train', 'valid');
set(l, 'FontSize', 15);
xlabel('\lambda', 'FontSize', 15);
ylabel('cross\_entropy', 'FontSize', 15);


subplot(2,2,4);
hold on;
title('mnist\_train\_small', 'FontSize', 15);
plot(lambda, frac_correct_train_lambda_small, 'LineWidth', 1.5);
plot(lambda, frac_correct_valid_lambda_small, 'LineWidth', 1.5);
l = legend('train', 'valid');
set(l, 'FontSize', 15);
xlabel('\lambda', 'FontSize', 15);
ylabel('frac\_correct', 'FontSize', 15);

