function [f,g] = Loss7(w,X,y,nHidden,nLabels)
% Loss7 
%
% Tianran Zhang, Dec. 5, 2017.

[nInstances,nVars] = size(X);
nH = length(nHidden);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);

for h = 2:nH
    hiddenWeights{h-1} = reshape(...
        w(offset+1:offset+nHidden(h-1)*nHidden(h)),...
        nHidden(h-1),nHidden(h));
    offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
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
end

% Compute Output
p = cell(1,nH);
for i = 1:nInstances
    p{1} = [rand(1, nHidden(1))>0.5];
   
    ip{1} = X(i,:) * inputWeights .* p{1};
    fp{1} = tanh(ip{1});
    for h = 2:nH
        p{h} = [rand(1, nHidden(h))>0.5];
        ip{h} = fp{h-1} * hiddenWeights{h-1} .* p{h};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end} * outputWeights;
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr;
        % Output Weights
        gOutput = (gOutput + fp{end}' * err) .* repmat(p{end}', 1, nLabels);
        
        if nH > 1
            % Last Layer of Hidden Weights
            backprop = err' * sech(ip{end}).^2 .* outputWeights';
            backprop = sum(backprop,1);
            gHidden{end} = (gHidden{end} + fp{end-1}' * backprop) .* ...
                repmat(p{end-1}', 1, length(p{end})) .* ...
                repmat(p{end}, length(p{end-1}), 1);
            
            % Other Hidden Layers
            for h = nH-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* ...
                    sech(ip{h+1}).^2;
                gHidden{h} = (gHidden{h} + fp{h}'*backprop) .* ...
                    (p{h}'*p{h+1});
            end
            
            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = (gInput + X(i,:)'*backprop) .* repmat(p{1}, nVars, 1);
        else
            % Input Weights
            gInput = gInput + X(i,:)' * (sech(ip{end}).^2 .* ...
                (outputWeights * err')');
           
            gInput = gInput .* repmat(p{1}, nVars, 1);
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
