function [f1, e] = convolution_f(c1,fWeights,hiddenWeights)
% CONVOLUTION_f computes the full-connected-layer values.
%
% Yuanbo Han, Dec. 9, 2017.

[nConv, nHidden] = size(hiddenWeights);
[c_row, c_col, ~] = size(c1);
f_row = size(c1,1) - size(fWeights,1) + 1;
f_col = size(c1,2) - size(fWeights,2) + 1;

f1 = zeros(f_row, f_col, nHidden);
e = zeros(c_row, c_col, nHidden);
for n = 1:nHidden
    count = 0;
    for m = 1:nConv
        count = count + c1(:,:,m) * hiddenWeights(m,n);
    end
    e(:,:,n) = count;
    f1(:,:,n) = conv2(e(:,:,n), ...
        rot90(fWeights(:,:,n),2), 'valid');
end

end