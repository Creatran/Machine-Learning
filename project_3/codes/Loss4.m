function [f,g] = Loss2(w, X, y, nHidden, nLabels, lambda)
% Loss2 Computes as much by matrix as possible to make the training fast.
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
        gOutput = gOutput + fp{end}' * err + lambda * outputWeights .* ...
            [zeros(1, size(outputWeights, 2)); ...
            ones(size(outputWeights, 1)-1, ...
            size(outputWeights, 2))];
        
        if nH > 1
            % Last Layer of Hidden Weights
            backprop = err' * sech(ip{end}).^2 .* outputWeights';
            backprop = sum(backprop,1);
            gHidden{end} = gHidden{end} + fp{end-1}' * backprop + ...
                lambda * hiddenWeights{end} .* ...
                [zeros(1, size(hiddenWeights{end}, 2)); ...
                ones(size(hiddenWeights{end}, 1)-1, ...
                size(hiddenWeights{end}, 2))];
            
            % Other Hidden Layers
            for h = nH-2:-1:1
                backprop = (backprop * hiddenWeights{h+1}') .* ...
                    sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop + ...
                    lambda * hiddenWeights{h} .* ...
                    [zeros(1, size(hiddenWeights{h}, 2)); ...
                    ones(size(hiddenWeights{h}, 1)-1, ...
                    size(hiddenWeights{h}, 2))];
                
            end
            
            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop + ...
                lambda * inputWeights .* ...
                [zeros(1, size(inputWeights, 2)); ...
                ones(size(inputWeights, 1)-1, ...
                size(inputWeights, 2))];
        else
            % Input Weights
            gInput = gInput + X(i,:)' * (sech(ip{end}).^2 .* ...
                (outputWeights * err')' ) + lambda * inputWeights .* ...
                [zeros(1, size(inputWeights, 2)); ...
                ones(size(inputWeights, 1)-1, ...
                size(inputWeights, 2))];   
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



function [f,g] = MLPclassificationLoss(w,X,y,nHidden,nLabels, lambda)
% g : the new weights
% f : the gradient

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
%offset：已用过weights个数
offset = nVars*nHidden(1);  
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),...
      nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

%gInput gHidden gOutput???
f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    yhat = fp{end}*outputWeights;  
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2*relativeErr;
        err1 = repmat(err, nHidden(end), 1);    
        fp1 = repmat(fp{end}', 1, nLabels);
        
        % Output Weights
        gOutput = gOutput + err1 .* fp1  + lambda * outputWeights .* ...
            [zeros(1, size(outputWeights, 2)); ...
            ones(size(outputWeights, 1)-1, size(outputWeights, 2))];      
       
        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop
            for c = 1:nLabels
                backprop(c,:) = err(c) * (sech(ip{end}).^2 .* ...
                    outputWeights(:,c)');
                gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:)+ ...
                    lambda * hiddenWeights{end};
            end
            backprop = sum(backprop,1);

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*sech(ip{h+1}).^2;
                gHidden{h} = gHidden{h} + fp{h}'*backprop + ...
                    lambda * hiddenWeights{end};
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*sech(ip{1}).^2;
            gInput = gInput + X(i,:)'*backprop + lambda*inputWeights;
        else
           % Input Weights
            gInput = gInput + X(i, :)' *(sech(ip{end}).^2.* ...
                sum((err1 .*outputWeights)' )) + lambda * inputWeights .* ...
                [zeros(1, size(inputWeights, 2)); ...
                ones(size(inputWeights, 1)-1, size(inputWeights, 2))];      
      
    end
    end    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
