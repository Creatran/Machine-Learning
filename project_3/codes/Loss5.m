function [f,g] = LossSoftmax(w,X,y,nHidden,nLabels)
% g : the new weights
% f : the gradient
%
% Tianran Zhang Dec.6th

[nInstances,nVars] = size(X);
nH = length(nHidden);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
    hiddenWeights{h-1} = reshape(w(offset+1 : offset+nHidden(h-1) * ...
        nHidden(h)), nHidden(h-1), nHidden(h));
    offset = offset + nHidden(h-1) * nHidden(h);
end
outputWeights = w(offset+1 : offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights, nHidden(end), nLabels);

f = 0;
ip = cell(1, nH);
fp = cell(1, nH);
if nargout > 1
    gInput = zeros(size(inputWeights));
    gHidden = cell(1, nH-1);
    for h = 2:nH
        gHidden{h-1} = zeros(size(hiddenWeights{h-1}));
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:) * inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    
    yhat1 = exp(fp{end}*outputWeights);
    yhat = yhat1/sum(yhat1);
    [~, y_true] = max(y(i, :));
    yhat_true = yhat(y_true);
    err = -log(yhat_true);
    f=f + err;
    
    if nargout > 1
        
        % Output Weights
        gOutput = gOutput - fp{end}' * (1 - yhat_true) * (y(i,:)==1);%
        if nH > 1
            % Last Layer of Hidden Weights
            backprop = err' * sech(ip{end}).^2 .* outputWeights';
            gHidden{end} = gHidden{end} + fp{end-1}' * sum(backprop,1);            
            backprop = sum(backprop,1);
        
            
            % Other Hidden Layers
            for h = nH-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}') .* ...
                    sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}' * backprop;
            end
            
            % Input Weights
            backprop = (backprop*hiddenWeights{1}') .* sech(ip{1}).^2;
            gInput = gInput + X(i,:)' * backprop;
        else
            % Input Weights
            gInput = gInput - (1 - yhat_true) * X(i,:)' * ...
                ( sech(ip{end}).^2 .* outputWeights(:, y(i,:)==1)' );
            
        end
    end
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:nH
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end