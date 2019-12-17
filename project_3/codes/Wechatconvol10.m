% Edited by Yuanbo Han, Dec. 9, 2017.
% Reference: http://blog.csdn.net/u010540396/article/details/52895074
tic;
load digits.mat
[n, d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and reshape X to be an array of n pixels.
[X,mu,sigma] = standardizeCols(X);
X = reshape(X',16,16,n);

% Apply the same transformation to the validation/test data.
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = reshape(Xvalid',16,16,t);
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = reshape(Xtest',16,16,t2);

% The number of neurons
nConv = 20;
nH = 100;

% Initialize bias.
cBias = randn(1, nConv);
fBias = randn(1, nH);
% Initialize convolution kernels.
cWeights = randn(5,5,nConv);
fWeights = randn(12,12,nH);
% Initialize weights for the full-connecting layer.
hiddenWeights = randn(nConv, nH);
outputWeights = randn(nH, nLabels);

% Train with stochastic gradient.
maxIter = 10000;
stepSize = 1e-3;
%tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/10)) == 0
        yhat = WechatCNN_predict(Xvalid, cWeights, fWeights, hiddenWeights, ...
            outputWeights, cBias, fBias);
        fprintf('Training iteration = %d, validation error = %f\n', ...
            iter-1, sum(yhat~=yvalid)/t);
       % toc;
        %tic;
    end
    
    i = ceil(rand*n);
    X1 = X(:,:,i);
    
    % Convolution layer
    c1 = zeros(12,12,nConv);
    for k = 1:nConv
        c1(:,:,k)=conv2(X1,rot90(cWeights(:,:,k),2),'valid');
        % apply tanh
        c1(:,:,k) = tanh(c1(:,:,k) + cBias(1,k));
    end
    
    % Full-connected layer
    [f1, e] = Wechatconvolution_f(c1, fWeights, hiddenWeights);
    % apply tanh
    f0 = zeros(1, nH);
    for h = 1:nH
        f0(1,h) = tanh(f1(:,:,h) + fBias(1,h));
    end
    
    % Output layer (Softmax)
    output = zeros(1, nLabels);
    for h = 1:nLabels
        output(1,h) = exp( f0*outputWeights(:,h) ) / ...
            sum( exp(f0*outputWeights) );
    end
    
    % Update weights, kernels and bias.
    [cWeights, fWeights, hiddenWeights, outputWeights, cBias, fBias] = ...
        WechatCNN_update(stepSize, y(i), X1, c1, f0, ...
        e, output, cWeights, fWeights, hiddenWeights, ...
        outputWeights, cBias, fBias);
end

yhat = WechatCNN_predict(Xtest, cWeights, fWeights, hiddenWeights, outputWeights, ...
    cBias, fBias);
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);

toc;