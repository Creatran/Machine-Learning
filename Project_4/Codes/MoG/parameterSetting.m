load digits;

logProb2 = zeros(10, 1);
logProb3 = zeros(10, 1);
t = 0;
for k = 0.1:0.1:1 
    [p2,mu2,vary2,logProbX2] = mogEM(train2, 2, 20, 0.01, 0, k);
    [p3,mu3,vary3,logProbX3] = mogEM(train3, 2, 20, 0.01, 0, k);
    
    t = t+1;
    logProb2(t) = logProbX2(20);
    logProb3(t) = logProbX3(20);
end

% print the log-prob against k
figure;
plot(0.1:0.1:1, logProb2);
xlabel('k', 'FontSize', 12);
ylabel('log-prob of X', 'FontSize', 12);
title('hand written 2', 'FontSize', 16);

figure;
plot(0.1:0.1:1, logProb3);
xlabel('k', 'FontSize', 12);
ylabel('log-prob of X', 'FontSize', 12);
title('hand written 3', 'FontSize', 16);