%Load Data
data = csvread('/home/prathamesh/OSL/E02/ex4dataset.txt'); % can also use load function
X = data(:,1:2);
y = data(:,3);
m = length(y);

%Normalizing features
fprintf('Normalizing Features ...\n');
[X mu sigma] = normalize(X);

%Add column of one to X
X = [ones(m,1), X];
% Initial settings for Cost function and gradient Descent
alpha = 0.01;
num_iters = 1500;
theta = zeros(3,1);

%Initial cost
initial_cost = cost(X, y, theta, m);
fprintf('\nInitial Cost before running gradient descent %f\n', initial_cost);

%Gradient Descent
fprintf('\nRunning gradient descent ...\n');
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

fprintf('\nCost after gradient descent:- %f\n', J_history(num_iters));

fprintf('\nTheta computed after gradient descent:\n');
fprintf(' %f \n', theta);
fprintf('\n');

%Predict price
predict = [1650 3];
predict_norm = (predict .- mu) ./ sigma;
predict_norm = [ones(1,1) predict_norm];
price = predict_norm * theta;

fprintf('\nPredicted price of a 1650 sq-ft, 3 room house: $%f\n', price);



%Observing the effect of different learning rates 
%on convergence of Gradient Descent
num_iters = 100;

alpha = [0.3; 0.1; 0.03; 0.01]; 

[_, J1] = gradientDescent(X, y, zeros(3,1), alpha(1), num_iters);
[_, J2] = gradientDescent(X, y, zeros(3,1), alpha(2), num_iters);
[_, J3] = gradientDescent(X, y, zeros(3,1), alpha(3), num_iters);
[_, J4] = gradientDescent(X, y, zeros(3,1), alpha(4), num_iters);


xlabel('Number of iterations');
ylabel('Cost J');
plot(1:numel(J1),[J1,J2,J3,J4],'LineWidth', 2); % numel -> no of elements
legend('alpha: 0.3', 'alpha: 0.1', 'alpha: 0.03', 'alpha: 0.01');              %