function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%   targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function

[n,m] = size(data);
y = sigmoid([data,ones(n,1)] * weights);

[f1, df1, y1] = logistic(weights, data, targets, hyperparameters);
weights(length(weights))=0;
h = hyperparameters.weight_regularization;
f = f1 + h/2 * sum(weights.^2);
df = df1 + h * weights;

end
