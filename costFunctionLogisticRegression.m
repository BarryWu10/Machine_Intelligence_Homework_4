function [J, grad] = costFunctionLogisticRegression(theta, X, y, lambda)
% costFunctionLogisticRegression Compute cost and gradient for logistic regression with regularization
%    [J, grad] = costFunctionLogisticRegression(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% number of training examples
n = length(y); 

% pre-allocate space for gradient
%grad = zeros(size(theta));
%dont need a vector of 0s

% Logistic Regression Cost Function
J = (1/n)*sum(-y.*(log(sigmoid(X*theta))) -(1-y).*log(1-(sigmoid(X*theta)))) + (lambda/(2*n))*sum(theta(2:end).^2);

theta_0 = theta;
theta_0(1) = 0;
grad = (1/n)*X'*(sigmoid(X*theta)-y) + (lambda/n)*theta_0;
end
