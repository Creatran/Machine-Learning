% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and variance
% for each class.

load mnist_train;
load mnist_test;

% Add your code here (it should be less than 10 lines)
[log_prior, class_mean, class_var] = train_nb(train_inputs, train_targets);
[Trainprediction, TrainAccuracy] = test_nb(train_inputs, train_targets, log_prior, class_mean, class_var);
[Testprediction, TestAccuracy] = test_nb(test_inputs, test_targets, log_prior, class_mean, class_var);

fprintf('training accuracy: %f\n', TrainAccuracy*100);
fprintf('test accuracy: %f\n', TestAccuracy*100);
plot_digits(class_mean);
plot_digits(class_var);
%fprintf('Test mu : %f  delta: %f\n', mean(Testprediction,1),var(Testprediction));

