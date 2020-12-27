function [J grad D2 D3 a2] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...%(25,401)
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...%(10,26)
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


##     part1 feedforward and j
##先算第二层layout
X = [ones(m,1) X];
z2 = Theta1 * X';
a2 = sigmoid(Theta1 * X');
%再算最后一层layout
a2 = [ones(1,m); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
%cost function without regulization
%y_range =[1:num_labels]';

% 循环法
%for example_m = 1:m
%  c = y(example_m);
%  y_1_0 = (y_range==c);
%  J_each_example = sum(-y_1_0.*log(z(:,example_m))-(1-y_1_0).*log(1-z(:,example_m)));
%  J = J + J_each_example/m;
%end
% -y_1_0.*log(sigmoid(a3)-(1-y_1_0).*log(1-sigmoid(z(:,example_m)))
% J = sum(J);

% 向量法
y_matrix = (eye(num_labels)(y,:))'; %例如 y =[2 3] 输出[0 1 0; 0 0 1] 需要转置成竖着的向量
J = 1/m * sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)));
regulization = lambda/(2*m) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );
J = J + regulization;

% Part2
% sigmoidgradient正确 a2正确
D3 = a3 - y_matrix; %[10 x 5000] 正确
D2 = (Theta2(:,2:end))' * D3 .* sigmoidGradient(z2); %[25 x 10] * [10 x 5000] .* [25 x 5000] = [25 x 5000]
Delta_2 = 1/m .* D3*(a2)'; % [10 x 5000] * [5000 x 26] =[10 x 26] 正确 without regulization

Delta_2_regulized = Delta_2(:,2:end) + lambda/m .* Theta2(:,2:end);
%Delta_2_0 = 1/m .* D3*(a2(1,:))';
%Delta_2_rest = 1/m .* D3*(a2(2:end,:))';
%Delta_2_regulized = [Delta_2_0 Delta_2_rest];

Delta_1 = 1/m .* D2* X; % [25 x 5000] * [5000 x 401] = [25 x 401] Remove d2_0 bias item.

Delta_1_regulized = Delta_1(:,2:end) + lambda/m .* Theta1(:,2:end);
%Delta_1_0 = 1/m .* D2*(X(1,:))';
%Delta_1_rest = 1/m .* D2*(X(2:end,:))';
%Delta_1_regulized = [Delta_1_0 Delta_1_rest];

Theta1_grad = [Delta_1(:,1) Delta_1_regulized];
Theta2_grad = [Delta_2(:,1) Delta_2_regulized];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
