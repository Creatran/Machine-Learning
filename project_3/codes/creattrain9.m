load digits.mat

%
tic
X_clock = zeros(size(X));
X_anticlock = zeros(size(X));
X_big = zeros(size(X));
X_small = zeros(size(X));

for i = 1:size(X,1)
    X1 = imrotate(reshape(X(i,:),16,16), 5, 'crop');
    X_clock(i,:) = X1(:);
    X1 = imrotate(reshape(X(i,:),16,16), -5, 'crop');
    X_anticlock(i,:) = X1(:);
    X1 = imresize(reshape(X(i,:),16,16), 1.1, 'OutputSize', [16,16]);
    X_big(i,:) = X1(:);
    X1 = imresize(reshape(X(i,:),16,16), 0.9, 'OutputSize', [16,16]);
    X_small(i,:) = X1(:);
end

X = [X; X_clock; X_anticlock; X_big; X_small];
y = repmat(y,5,1);
toc

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
nHidden = [10];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;% * 3;
%momentumStrength = 0.9;
%delta = 0;
%lambda = 0.03;
funObj = @(w,i)Loss2(w,X(i,:),yExpanded(i,:),nHidden,nLabels);

tic
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = Predict2(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [~,g] = funObj(w,i);
    %    delta = stepSize * g - momentumStrength * delta;
    %    w = w - delta;
    w = w - stepSize * g;
end

% Evaluate test error
yhat = Predict2(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
toc