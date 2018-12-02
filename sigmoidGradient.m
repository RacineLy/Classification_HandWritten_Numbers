function g = sigmoidGradient(z)

g = zeros(size(z));
vartemp = 1./(1 + exp(-z));
g = vartemp.*(1 - vartemp);


end
