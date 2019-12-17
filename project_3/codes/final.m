% Add l2 regularization
%
% Tianran Zhang, Dec. 5, 2017.

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

% Choose network structure
nHidden = [100];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
maxIter = 100000;
w = randn(nParams,1);

% Train with stochastic gradient
stepSize = 1e-3;

funObj = @(w,i)Loss_final(w,X(i,:),yExpanded(i,:),nHidden,nLabels, 0.03);

tic;
w0 = 0;
a0 = 1;
a1 = 1;
for iter = 1:maxIter    
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = Predict5(w,Xvalid,nHidden,nLabels);
        a2 = sum(yhat~=yvalid)/t;
        fprintf('Training iteration = %d, validation error = %f\n',iter-1, a2);
        
        %%early stop       
        if (a2 > a1 )&(a1 > a0) 
            break;
        else
            a0 = a1;
            a1 = a2;
        end        
    end
    
    i = ceil(rand*n);
    [~, g] = funObj(w,i); 
    w = w - stepSize*g + 0.9 * (w - w0);
    w0 = w;
end
toc;

% Evaluate test error
yhat = Predict5(w, Xtest, nHidden, nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
