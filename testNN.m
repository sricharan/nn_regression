function testNN(nn_params, input_layer_size, hidden_layer_size, num_labels, lambda)


data = load("validation_data.txt");

X = data(:,[1,2]);

y = data(:,[3,4,5,6,7]);

X = featureNormalize(X);

y = featureNormalize(y);

data_matrix = [X y];

A = zeros(rows(data_matrix),1);

%lambda = 1;

%for i = 1:rows(X)

  J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda);
  
% A(i) = J;

%end 

save("-ascii", "test_cost.txt","A");

disp("cost=");

disp(J);
