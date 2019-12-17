function [f, g, g1] = Loss10(w1, w2, X1, y, nHidden, nLabels)
% Loss2 Computes as much by matrix as possible to make the training fast.
%
% Tianran Zhang, Dec. 5, 2017.

X1 = reshape(X1, 16, 16);
X = zeros(12, 12);
for i =1 : 12
    for j = 1:12
        X(i, j) = sum(sum(X1(i : i+4, j: j+4) .* w1));
    end
end

X = reshape(X, 1, 144);
[nInstances,nVars] = size(X);
nH = length(nHidden);

% Form Weights
inputWeights = reshape(w2(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:nH
    hiddenWeights{h-1} = reshape(...
        w2(offset+1:offset+nHidden(h-1)*nHidden(h)),...
        nHidden(h-1),nHidden(h));
    offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w2(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
ip = cell(1, nH);
fp = cell(1, nH);
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:nH
        gHidden{h-1} = zeros(size(hiddenWeights{h-1}));
    end
    gOutput = zeros(size(outputWeights));
    g1 = zeros(25, 1);
end

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:) * inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:nH
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end} * outputWeights;
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr;
        % Output Weights
        gOutput = gOutput + fp{end}' * err;
        
        if nH > 1
            % Last Layer of Hidden Weights
            backprop = err' * sech(ip{end}).^2 .* outputWeights';
            backprop = sum(backprop,1);
            gHidden{end} = gHidden{end}; 
            
            % Other Hidden Layers
            for h = nH-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* ...
                    sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end
            
            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop;
        else
            % Input Weights
            gInput = gInput + X(i,:)' * (sech(ip{end}).^2 .* ...
                (outputWeights * err')' );
            g0 = findp(X, w1);
            g1 = g1 + g0' * inputWeights * (sech(ip{end}).^2 .* ...
                (outputWeights * err')' );
            g1 = reshape(g1, 5,5);
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

end
