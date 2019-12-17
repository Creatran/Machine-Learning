function [w1] = new_w(w,X,y,nHidden,nLabels)
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

ip = cell(1, nH);
fp = cell(1, nH);

% Compute w1
w1 = w;
for i = 1:nInstances
    ip{1} = X(i,:) * inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:nH
        ip{h} = fp{h-1} * hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    outputWeights = pinv(fp{end}' * fp{end}) * fp{end}' * y(i,:);
    w1(offset+1:offset+nHidden(end)*nLabels) = outputWeights(:);
end

