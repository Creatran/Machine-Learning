function [yhat] = CNN_predict(X, cWeights, fWeights, hiddenWeights, ...
    outputWeights, cBias, fBias)

nInstances = size(X, 3);
yhat = zeros(nInstances, 1);
nConv = size(cWeights, 3);
[nHidden, nLabels] = size(outputWeights);
c1 = size(X,1) - size(cWeights,1) + 1;
c2 = size(X,2) - size(cWeights,2) + 1;

for i = 1:nInstances
    train_data = X(:,:,i);
    
    % Convolution layer
    c0 = zeros(c1, c2, nConv);
    for k = 1:nConv
        c0(:,:,k) = conv2(train_data,rot90(cWeights(:,:,k),2),'valid');
        % apply tanh
        c0(:,:,k) = tanh(c0(:,:,k) + cBias(1,k));
    end
    
    % Full-connected layer
    [f1,~] = Wechatconvolution_f(c0, fWeights, hiddenWeights);
    % apply tanh
    f = zeros(1, nHidden);
    for h = 1:nHidden
        f(1,h) = tanh(f1(:,:,h) + fBias(1,h));
    end
    
    % Output layer (Softmax)
    output = zeros(1, nLabels);
    for h = 1:nLabels
        output(1,h) = exp( f * outputWeights(:,h) ) / ...
            sum( exp(f * outputWeights) );
    end
    [~, yhat(i)] = max(output);
end

end