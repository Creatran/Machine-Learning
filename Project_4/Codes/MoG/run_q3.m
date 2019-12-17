load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

[p2,mu2,vary2,logProbX2] = mogEM(train2, 2, 20, 0.01, 0);
[p3,mu3,vary3,logProbX3] = mogEM(train3, 2, 20, 0.01, 0);

% plot mean and variance vectors of each data
plot_digit(mu2, 1, 2);
plot_digit(vary2, 1, 2);
plot_digit(mu3, 1, 2);
plot_digit(vary3, 1, 2);

% print the proportion for the clusters within each model
fprintf('proportion of train2:\n');
disp(p2);
fprintf('proportion of train3:\n');
disp(p3);

% Provide logP(TrainingData) for each model
fprintf('logP(Train2): %f\n', logProbX2(end));
fprintf('logP(Train3): %f\n', logProbX3(end));
