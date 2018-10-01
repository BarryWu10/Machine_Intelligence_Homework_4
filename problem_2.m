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
plotDecisionBoundary(theta_0, Xdata, y, degree);

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

title_string = ['{\lambda}=', num2str(lambda_0),', Accuracy=', num2str(accurracy_0*100),'%'];
title(title_string,'fontsize',14);
xlabel('Microchip Test 1','fontsize',12)
ylabel('Microchip Test 2','fontsize',12)
legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

print -dpng hwk4_problem2_lambda_0_plot.png

%% lambda = 1
lambda_1 = 1;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_1, J_1, exit_flag_1] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_1)), initial_theta, options);
plotDecisionBoundary(theta_1, Xdata, y, degree);

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

title_string = ['{\lambda}=', num2str(lambda_1),', Accuracy=', num2str(accurracy_1 * 100),'%'];
title(title_string,'fontsize',14);
xlabel('Microchip Test 1','fontsize',12)
ylabel('Microchip Test 2','fontsize',12)
legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

print -dpng hwk4_problem2_lambda_1_plot.png

%% lambda = 10
lambda_10 = 10;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_10, J_10, exit_flag_10] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_10)), initial_theta, options);
plotDecisionBoundary(theta_10, Xdata, y, degree);

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

title_string = ['{\lambda}=', num2str(lambda_10),', Accuracy=', num2str(accurracy_10 * 100),'%'];
title(title_string,'fontsize',14);
xlabel('Microchip Test 1','fontsize',12)
ylabel('Microchip Test 2','fontsize',12)
legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

print -dpng hwk4_problem2_lambda_10_plot.png

%% lambda = 100
lambda_100 = 100;
% Specifying function with the @(t) allows fminunc to call our costFunction
% The t is an input argument, in this case initial_theta
[theta_100, J_100, exit_flag_100] = ...
    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda_100)), initial_theta, options);
plotDecisionBoundary(theta_100, Xdata, y, degree);

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

title_string = ['{\lambda}=', num2str(lambda_100),', Accuracy=', num2str(accurracy_100 * 100),'%'];
title(title_string,'fontsize',14);
xlabel('Microchip Test 1','fontsize',12)
ylabel('Microchip Test 2','fontsize',12)
legend('Admitted', 'Not admitted', 'Decision Boundary','location', 'best')

print -dpng hwk4_problem2_lambda_100_plot.png