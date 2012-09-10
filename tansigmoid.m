function g = tansigmoid(z)
%TANSIGMOID Compute sigmoid functoon
%   J = TANSIGMOID(z) computes the sigmoid of z.

g = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));
end
