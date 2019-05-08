clc;

% load small_features.mat;
% load small_labels.mat;
% load medium_features.mat;
% load medium_labels.mat;
load big_features.mat;
% load big_labels.mat;
load big_labels.mat;

% small data set
% X_small = features;
% Y_small = labels;

% medium data set
X = features;
Y = labels;


[m,d] = size(X);

% random split the data (80% for training, 20% for testing)
r = randperm(m);
t = round(m*0.7);
X_train = X(r(1:t),:);
Y_train = Y(r(1:t));
X_test = X(r(t+1:end),:);
Y_test = Y(r(t+1:end));

% SVM without kernel
svm = fitcsvm(X_train,Y_train);

% SVM with RBF kernel
rbf_svm = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

Y_hat = predict(rbf_svm, X_test);

numErrors = sum(abs(Y_hat - Y_test));
errorRate = numErrors / size(Y_test,1);
disp(errorRate);
error = Y_hat - Y_test;
e1 = length(find(error==-1));
e2 = length(find(error==1));