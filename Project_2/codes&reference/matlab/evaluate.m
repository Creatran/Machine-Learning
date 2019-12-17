function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entartropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function

[n, m] = size(targets);
ce = -mean([targets==1] .* log(y)+[targets==0] .* (log(1-y)));
frac_correct = sum([(targets-0.5).*(y-0.5)>0])/n;
    
end
