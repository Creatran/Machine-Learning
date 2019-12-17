% find_nHidden tried the number of units in one layer from 1 to 
% (input numbers + output numbers). Find out the error of each number of 
% units. Draw a plot of error-units and work out the best number of units.
%
% Tianran Zhang, Dec. 1, 2017.

load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

errors = [];
num1 = 1 : (d+nLabels);

for num = num1
    % Choose network structure
    nHidden = [num];
    
    % Count number of parameters and initialize weights 'w'
    nParams = d*nHidden(1) + nHidden(end) * nLabels;
    w = randn(nParams,1);
    
    % Train with stochastic gradient    
    maxIter = 10000;
    stepSize = 1e-3;
    funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),...
        nHidden,nLabels);
    
    k = 0;
    for iter = 1:maxIter
        if mod(iter-1,round(maxIter/20)) == 0
            yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
            k = k+1;
            er (k) = sum(yhat~=yvalid)/t;
        end
        i = ceil(rand*n);
        [~, g] = funObj(w,i);
        w = w - stepSize*g;
    end
    
    errors(num) = mean(er);
end
toc;

figure;
plot(num1, errors);
xlabel('num of units');
ylabel('validation error');