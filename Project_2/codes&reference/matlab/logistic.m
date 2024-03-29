function [f, df, y] = logistic(weights, data, targets, hyperparameters)
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
%	targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value?i.e. negative log likelihood).
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function

[n, m] = size(data);
x = [data , ones(n,1)];
w = x*weights;
%f = - tagrets' * (x * weights)- sum(log(sigmoid(x * weights)));
f = -sum(x * weights .* targets)+ sum(log(1+ exp(x * weights)));
df = -x' * (targets-sigmoid(x * weights));
y = logistic_predict(weights, data);

end
