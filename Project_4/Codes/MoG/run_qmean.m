load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

[p,mu,vary,logProbX] = mogEM(x, 20, 20, 0.01, 0);

% Provide logP(TrainingData) for each model
fprintf('logP(x): %f\n', logProbX(end));

