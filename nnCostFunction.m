function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layers neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];                    % Design matrix

% Layer 1 to Layer 2
z2 = Theta1*X';                       % Argument Layer 2
a2 = sigmoid(z2);                     % Activation Layer 2
a2 = [ones(1,size(a2,2));a2];         % Add bias unit

% Layer 2 to Layer 3
z3 = Theta2*a2;                       % Argument Layer 3
a3 = sigmoid(z3);                     % Activation Layer 3
hk = a3;                              % Output Layer

yk = zeros(num_labels,m);             % Compute yk

for k = 1:m
  yk(y(k),k) = 1;
endfor

J = (1/m)*sum(sum(-yk.*log(hk) - (1-yk).*log(1-hk)));   % Compute Cost without Regularization

Temp1 = (Theta1(:,(2:end))).^2;
Temp2 = (Theta2(:,(2:end))).^2;

RegTerm = sum(sum(Temp1)) + sum(sum(Temp2));            % Regularization term

J = J + (lambda/(2*m))*RegTerm;                         % Cost function with Regularization

% Cost Function Gradient using Back propagation

for t = 1:m
  
  % STEP 1
  a1 = X(t,:)';
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  hk = a3;
  
  % STEP 2
  var = ([1:num_labels] == y(t))';
  delta3 = a3 - var;
  
  % STEP 3
  delta2 = (Theta2)'*delta3.*sigmoidGradient([1;z2]);
  delta2 = delta2(2:end);
  
  % STEP 4
  Theta1_grad = Theta1_grad + delta2*(a1)';
  Theta2_grad = Theta2_grad + delta3*(a2)';
  
endfor

% -------------------------------------------------------------

% =========================================================================
% for j = 0
Theta1_grad(:,1) = (1/m)*Theta1_grad(:,1);
Theta2_grad(:,1) = (1/m)*Theta2_grad(:,1);

% for j >= 1
Theta1_grad(:,(2:end)) = (1/m)*Theta1_grad(:,(2:end)) + (lambda/m)*Theta1_grad(:,(2:end));
Theta2_grad(:,(2:end)) = (1/m)*Theta2_grad(:,(2:end)) + (lambda/m)*Theta2_grad(:,(2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
