function [f,g] = MLPclassificationLoss(w,X,y,nHidden,nLabels)
% g : the gradient
% f : err

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
%offset：已用过weights个数
offset = nVars*nHidden(1);  
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

%gInput gHidden gOutput???
f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
   
    yhat = fp{end}*outputWeights;
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = relativeErr*2;

        % Output Weights
        for c = 1:nLabels
            gOutput(:,c) = gOutput(:,c) + err(c)*fp{end}';
        end
        
           % Input Weights
        for c = 1:nLabels
                gInput = gInput + err(c)*X(i,:)'*(sech(ip{end}).^2.*outputWeights(:,c)');
        end
    end    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end
