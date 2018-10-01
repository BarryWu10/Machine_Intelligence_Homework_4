clear ; close all; clc
% Load Data (from Andrew Ng Machine Learning online MOOC)
% The first two columns contains the X values and the third column
% contains the label (y).
data = load('ex2data2.txt'); %data is 118x3
X = data(:, [1, 2]); y = data(:, 3);
% The data points that are not
% linearly separable. However, you would still like to use logistic
% regression to classify the data points.
%
% To do so, you introduce more features to use -- in particular, you add
% polynomial features to our data matrix (similar to polynomial
% regression).
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
degree=6; %degree of polynomial allowed
Xdata = mapFeature(X(:,1), X(:,2),degree);
% Initialize fitting parameters
initial_theta = zeros(size(Xdata, 2), 1);
% Set regularization parameter lambda to 1 (you should vary this)
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize

%% lambda = 0
lambda_0 = 0;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_0, J_0, exit_flag_0] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_0)), initial_theta, options);

TPTN_0 = 0;
TPTNFPFN_0 = 0;

y_hat_0 = sigmoid(Xdata*theta_0);
for index_y_hat_0 = 1:size(y_hat_0)
    if y_hat_0(index_y_hat_0) >= 0.5
        y_hat_0(index_y_hat_0) = 1;
    else
        y_hat_0(index_y_hat_0) = 0;
    end
    if y_hat_0(index_y_hat_0) == y(index_y_hat_0)
        TPTN_0 = TPTN_0 + 1;
        TPTNFPFN_0 = TPTNFPFN_0 + 1;
    else
        TPTNFPFN_0 = TPTNFPFN_0 + 1;
    end
end

accurracy_0 = TPTN_0/TPTNFPFN_0;

% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for
%each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds=5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
method='LogisticRegression';
%lambda=1
for i = 1:numberOfFolds
TestIndex_0 = find(CVindex == i);
TrainIndex_0 = find(CVindex ~= i);
TrainDataCV_0 = Xdata(TrainIndex_0,:);
TrainDataGT_0 = y(TrainIndex_0);
TestDataCV_0 = Xdata(TestIndex_0,:);
TestDataGT_0 = y(TestIndex_0);
%
%build the model using TrainDataCV and TrainDataGT
%test the built model using TestDataCV
%
switch method
    case 'LogisticRegression'
    % for Logistic Regression, we need to solve for theta
    % Insert code here to solve for theta...
    [theta_0_train, J_0_train, exit_flag_0_train] = ...
        fminunc(@(t)(costFunctionLogisticRegression(t, TrainDataCV_0, TrainDataGT_0, lambda_0)), initial_theta, options);
    %plotDecisionBoundary(theta_0_train, TrainDataCV_0, TrainDataGT_0, degree);
    % Using TestDataCV, compute testing set prediction using
    % the model created
    % for Logistic Regression, the model is theta
    % Insert code here to see how well theta works...
    TestDataPred_0 = TestDataCV_0 * theta_0_train > 0.5;
    case 'KNN'
        disp('KNN not implemented yet')
    otherwise
        error('Unknown classification method')
end
predictionLabels_0(TestIndex_0,:) = double(TestDataPred_0);
end
confusionMatrix_0 = confusionmat(y,predictionLabels_0);
accuracy_0_cf = sum(diag(confusionMatrix_0))/sum(sum(confusionMatrix_0));
fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method,lambda_0,accuracy_0_cf*100));
%fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix_0);
for i=1:r
for j=1:r
fprintf('%6d ',confusionMatrix_0(i,j));
end
fprintf('\n');
end

figure
title_string_0 = ['{\lambda}=', num2str(lambda_0),', Accuracy=', num2str(accuracy_0_cf * 100),'%'];
confusionchart(confusionMatrix_0, 'Title', title_string_0);
print -dpng hwk4_problem3_lambda_0_plot.png

%figure
%plotDecisionBoundary(theta_0, Xdata, y, degree);
%title_string = ['{\lambda}=', num2str(lambda_0),', Accuracy=', num2str(accurracy_0*100),'%'];
%title(title_string,'fontsize',14);
%xlabel('Microchip Test 1','fontsize',12)
%ylabel('Microchip Test 2','fontsize',12)
%legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

%% lambda = 1
lambda_1 = 1;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_1, J_1, exit_flag_1] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_1)), initial_theta, options);
%plotDecisionBoundary(theta_1, Xdata, y, degree);

TPTN_1 = 0;
TPTNFPFN_1 = 0;

y_hat_1 = sigmoid(Xdata*theta_1);
for index_y_hat_1 = 1:size(y_hat_1)
    if y_hat_1(index_y_hat_1) >= 0.5
        y_hat_1(index_y_hat_1) = 1;
    else
        y_hat_1(index_y_hat_1) = 0;
    end
    if y_hat_1(index_y_hat_1) == y(index_y_hat_1)
        TPTN_1 = TPTN_1 + 1;
        TPTNFPFN_1 = TPTNFPFN_1 + 1;
    else
        TPTNFPFN_1 = TPTNFPFN_1 + 1;
    end
end

accurracy_1 = TPTN_1/TPTNFPFN_1;

% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for
%each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds=5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
method='LogisticRegression';
%lambda=1
for i = 1:numberOfFolds
TestIndex_1 = find(CVindex == i);
TrainIndex_1 = find(CVindex ~= i);
TrainDataCV_1 = Xdata(TrainIndex_1,:);
TrainDataGT_1 = y(TrainIndex_1);
TestDataCV_1 = Xdata(TestIndex_1,:);
TestDataGT_1 = y(TestIndex_1);
%
%build the model using TrainDataCV and TrainDataGT
%test the built model using TestDataCV
%
switch method
    case 'LogisticRegression'
    % for Logistic Regression, we need to solve for theta
    % Insert code here to solve for theta...
    [theta_1_train, J_1_train, exit_flag_1_train] = ...
        fminunc(@(t)(costFunctionLogisticRegression(t,TrainDataCV_1, TrainDataGT_1, lambda_1)), initial_theta, options);
    %plotDecisionBoundary(theta_1_train, TrainDataCV_1, TrainDataGT_1, degree);
    % Using TestDataCV, compute testing set prediction using
    % the model created
    % for Logistic Regression, the model is theta
    % Insert code here to see how well theta works...
    TestDataPred_1 = TestDataCV_1 * theta_1_train > 0.5;
    case 'KNN'
        disp('KNN not implemented yet')
    otherwise
        error('Unknown classification method')
end
predictionLabels_1(TestIndex_1,:) = double(TestDataPred_1);
end
confusionMatrix_1 = confusionmat(y,predictionLabels_1);
accuracy_1_cf = sum(diag(confusionMatrix_1))/sum(sum(confusionMatrix_1));
fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method,lambda_1,accuracy_1_cf*100));
%fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix_1);
for i=1:r
for j=1:r
fprintf('%6d ',confusionMatrix_1(i,j));
end
fprintf('\n');
end

figure
title_string_1 = ['{\lambda}=', num2str(lambda_1),', Accuracy=', num2str(accuracy_1_cf * 100),'%'];
confusionchart(confusionMatrix_1, 'Title', title_string_1);
print -dpng hwk4_problem3_lambda_1_plot.png

%title_string = ['{\lambda}=', num2str(lambda_1),', Accuracy=', num2str(accurracy_1 * 100),'%'];
%title(title_string,'fontsize',14);
%xlabel('Microchip Test 1','fontsize',12)
%ylabel('Microchip Test 2','fontsize',12)
%legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

%print -dpng hwk4_problem3_lambda_1_plot.png

%% lambda = 10
lambda_10 = 10;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_10, J_10, exit_flag_10] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_10)), initial_theta, options);
%plotDecisionBoundary(theta_10, Xdata, y, degree);

TPTN_10 = 0;
TPTNFPFN_10 = 0;

y_hat_10 = sigmoid(Xdata*theta_10);
for index_y_hat_10 = 1:size(y_hat_10)
    if y_hat_10(index_y_hat_10) >= 0.5
        y_hat_10(index_y_hat_10) = 1;
    else
        y_hat_10(index_y_hat_10) = 0;
    end
    if y_hat_10(index_y_hat_10) == y(index_y_hat_10)
        TPTN_10 = TPTN_10 + 1;
        TPTNFPFN_10 = TPTNFPFN_10 + 1;
    else
        TPTNFPFN_10 = TPTNFPFN_10 + 1;
    end
end

accurracy_10 = TPTN_10/TPTNFPFN_10;

% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for
%each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds=5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
method='LogisticRegression';
%lambda=1
for i = 1:numberOfFolds
TestIndex_10 = find(CVindex == i);
TrainIndex_10 = find(CVindex ~= i);
TrainDataCV_10 = Xdata(TrainIndex_10,:);
TrainDataGT_10 = y(TrainIndex_10);
TestDataCV_10 = Xdata(TestIndex_10,:);
TestDataGT_10 = y(TestIndex_10);
%
%build the model using TrainDataCV and TrainDataGT
%test the built model using TestDataCV
%
switch method
    case 'LogisticRegression'
    % for Logistic Regression, we need to solve for theta
    % Insert code here to solve for theta...
    [theta_10_train, J_10_train, exit_flag_10_train] = ...
        fminunc(@(t)(costFunctionLogisticRegression(t, TrainDataCV_10, TrainDataGT_10, lambda_10)), initial_theta, options);
    %plotDecisionBoundary(theta_1_train, TrainDataCV_1, TrainDataGT_1, degree);
    % Using TestDataCV, compute testing set prediction using
    % the model created
    % for Logistic Regression, the model is theta
    % Insert code here to see how well theta works...
    TestDataPred_10 = TestDataCV_10 * theta_10_train > 0.5;
    case 'KNN'
        disp('KNN not implemented yet')
    otherwise
        error('Unknown classification method')
end
predictionLabels_10(TestIndex_10,:) = double(TestDataPred_10);
end
confusionMatrix_10 = confusionmat(y,predictionLabels_10);
accuracy_10_cf = sum(diag(confusionMatrix_10))/sum(sum(confusionMatrix_10));
fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method,lambda_10,accuracy_10_cf*100));
%fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix_10);
for i=1:r
for j=1:r
fprintf('%6d ',confusionMatrix_10(i,j));
end
fprintf('\n');
end

figure
title_string_10 = ['{\lambda}=', num2str(lambda_10),', Accuracy=', num2str(accuracy_10_cf * 100),'%'];
confusionchart(confusionMatrix_10, 'Title', title_string_10);
print -dpng hwk4_problem3_lambda_10_plot.png


%title_string = ['{\lambda}=', num2str(lambda_10),', Accuracy=', num2str(accurracy_10 * 100),'%'];
%title(title_string,'fontsize',14);
%xlabel('Microchip Test 1','fontsize',12)
%ylabel('Microchip Test 2','fontsize',12)
%legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

%print -dpng hwk4_problem3_lambda_10_plot.png

%% lambda = 100
lambda_100 = 100;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_100, J_100, exit_flag_100] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_100)), initial_theta, options);
%plotDecisionBoundary(theta_100, Xdata, y, degree);

TPTN_100 = 0;
TPTNFPFN_100 = 0;

y_hat_100 = sigmoid(Xdata*theta_100);
for index_y_hat_100 = 1:size(y_hat_100)
    if y_hat_100(index_y_hat_100) >= 0.5
        y_hat_100(index_y_hat_100) = 1;
    else
        y_hat_100(index_y_hat_100) = 0;
    end
    if y_hat_100(index_y_hat_100) == y(index_y_hat_100)
        TPTN_100 = TPTN_100 + 1;
        TPTNFPFN_100 = TPTNFPFN_100 + 1;
    else
        TPTNFPFN_100 = TPTNFPFN_100 + 1;
    end
end

accurracy_100 = TPTN_100/TPTNFPFN_100;

% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for
%each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds=5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
method='LogisticRegression';
%lambda=1
for i = 1:numberOfFolds
TestIndex_100 = find(CVindex == i);
TrainIndex_100 = find(CVindex ~= i);
TrainDataCV_100 = Xdata(TrainIndex_100,:);
TrainDataGT_100 = y(TrainIndex_100);
TestDataCV_100 = Xdata(TestIndex_100,:);
TestDataGT_100 = y(TestIndex_100);
%
%build the model using TrainDataCV and TrainDataGT
%test the built model using TestDataCV
%
switch method
    case 'LogisticRegression'
    % for Logistic Regression, we need to solve for theta
    % Insert code here to solve for theta...
    [theta_100_train, J_100_train, exit_flag_100_train] = ...
        fminunc(@(t)(costFunctionLogisticRegression(t, TrainDataCV_100, TrainDataGT_100, lambda_100)), initial_theta, options);
    %plotDecisionBoundary(theta_1_train, TrainDataCV_1, TrainDataGT_1, degree);
    % Using TestDataCV, compute testing set prediction using
    % the model created
    % for Logistic Regression, the model is theta
    % Insert code here to see how well theta works...
    TestDataPred_100 = TestDataCV_100 * theta_100_train > 0.5;
    case 'KNN'
        disp('KNN not implemented yet')
    otherwise
        error('Unknown classification method')
end
predictionLabels_100(TestIndex_100,:) = double(TestDataPred_100);
end
confusionMatrix_100 = confusionmat(y,predictionLabels_100);
accuracy_100_cf = sum(diag(confusionMatrix_100))/sum(sum(confusionMatrix_100));
fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method,lambda_100,accuracy_100_cf*100));
%fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix_100);
for i=1:r
for j=1:r
fprintf('%6d ',confusionMatrix_100(i,j));
end
fprintf('\n');
end

figure
title_string_100 = ['{\lambda}=', num2str(lambda_100),', Accuracy=', num2str(accuracy_100_cf * 100),'%'];
confusionchart(confusionMatrix_100, 'Title', title_string_100);
print -dpng hwk4_problem3_lambda_100_plot.png

%title_string = ['{\lambda}=', num2str(lambda_100),', Accuracy=', num2str(accurracy_100 * 100),'%'];
%title(title_string,'fontsize',14);
%xlabel('Microchip Test 1','fontsize',12)
%ylabel('Microchip Test 2','fontsize',12)
%legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

%print -dpng hwk4_problem3_lambda_100_plot.png