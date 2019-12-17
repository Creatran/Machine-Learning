load digits;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];

for i = 1 : 4
    K = numComponent(i);
    % Train a MoG model with K components for digit 2
    %-------------------- Add your code here --------------------------------
    [p2,mu2,vary2,logProbX2] = mogEM(train2, K, 4, 0.01, 0);
    
    % Train a MoG model with K components for digit 3
    %-------------------- Add your code here ------------------------------
    [p3,mu3,vary3,logProbX3] = mogEM(train2, K, 4, 0.01, 0);
    
    
    % Caculate the probability P(d=1|x) and P(d=2|x),
    % classify examples, and compute the error rate
    % Hints: you may want to use mogLogProb function
    %-------------------- Add your code here ------------------------------
    %Training Part
    logProb_train2 = mogLogProb(p2, mu2, vary2, train2);
    logProb_train3 = mogLogProb(p3, mu3, vary3, train3);
    
    errorTrain(i) = 1 - (sum(logProb_train2 > 0.5) + ...
        sum(logProb_train3 > 0.5))/(size(train2, 2)/2 + size(train3, 2));
    
    
    %Valid Part
    logProb_valid2 = mogLogProb(p2, mu2, vary2, valid2);
    logProb_valid3 = mogLogProb(p3, mu3, vary3, valid3);
    errorValidation(i) = 1 - (sum(logProb_valid2 > 0.5) + ...
        sum(logProb_valid3 > 0.5))/(size(valid2, 2)/2 + size(valid3, 2));
    
    %Test Part
    logProb_test2 = mogLogProb(p2, mu2, vary2, test2);
    logProb_test3 = mogLogProb(p3, mu3, vary3, test3);
    errorTest(i) = 1 - (sum(logProb_test2 > 0.5) + ...
        sum(logProb_test3 > 0.5))/(size(test2, 2)/2 + size(test3, 2));
    
end

% Plot the error rate
%-------------------- Add your code here --------------------------------

figure;
hold on;
plot(numComponent, errorTrain, 'LineWidth', 4);
plot(numComponent, errorValidation, 'LineWidth', 4);
plot(numComponent, errorTest, 'LineWidth', 4);
legend({'Error rate on training set', 'Error rate on training set',...
    'Error rate on test set'}, 'FontSize', 12);
