function [f,g] = Loss6(w,X,y,nHidden,nLabels)
% Loss2 Computes as much by matrix as possible to make the training fast.
%
% Tianran Zhang, Dec. 5, 2017.

[nInstances,nVars] = size(X);
nH = length(nHidden);

% Form Weights
inputWeights = reshape(w(1:nVars*(nHidden(1)-1)), nVars, nHidden(1)-1);
offset = nVars * (nHidden(1)-1);
for h = 2:nH
    hiddenWeights{h-1} = reshape(...
        w(offset+1 : offset+nHidden(h-1)*(nHidden(h)-1)),...
        nHidden(h-1) , nHidden(h)-1);
    offset = offset + nHidden(h-1)*(nHidden(h)-1);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights, nHidden(end), nLabels);

f = 0;
ip = cell(1, nH);
fp = cell(1, nH);
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:nH
        gHidden{h-1} = zeros(size(hiddenWeights{h-1}));
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    ip{1} = [X(i,:) * inputWeights, 1];
    fp{1} = tanh(ip{1});
    for h = 2:nH
        ip{h} = [fp{h-1} * hiddenWeights{h-1}, 1];
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
            g1 = fp{end-1}' * backprop;
            gHidden{end} = gHidden{end} + g1(: , 1: nHidden(end)-1);
            
            % Other Hidden Layers
            for h = nH-2 : -1 : 1
                g1 = (backprop * hiddenWeights{h+1}') .* ...
                    sech(ip{h+1}).^2;
                backprop = g1(:, 1: nHidden(h)-1);
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end
            
            % Input Weights
            g1 = (backprop(:, 1:nHidden(2)-1)*hiddenWeights{1}').*sech(ip{1}).^2;
            backprop = g1(:, 1:nHidden(1)-1);
            gInput = gInput + X(i,:)'*backprop;
        else
            % Input Weights
            gx = X(i,:)' * (sech(ip{end}).^2 .* ...
                (outputWeights * err')' );
            gInput = gInput + gx(: , 1: nHidden(1)-1);
        end
    end
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*(nHidden(1)-1)) = gInput(:);
    offset = nVars*(nHidden(1)-1);
    for h = 2:nH
        g(offset+1:offset+nHidden(h-1)*(nHidden(h)-1)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*(nHidden(h)-1);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
