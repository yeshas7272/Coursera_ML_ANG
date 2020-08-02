function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hTheta_X  = sigmoid(X * theta);

J = -(1/m) * sum((y .* log(hTheta_X)) + ((1 - y) .* log(1 - hTheta_X))) + ((lambda / (2 * m)) * sum(theta(2:size(theta,1)) .^ 2));
Xb = X' ;
grad(1) = (1/m) * Xb(1,:) * (hTheta_X - y );

grad(2:size(theta,1)) = (1/m) * (Xb(2:size(Xb,1),:) * (hTheta_X - y ) + (lambda) * theta(2:size(theta,1),:));

% =============================================================

end
