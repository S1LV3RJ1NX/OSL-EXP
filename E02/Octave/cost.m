function J = cost(X, y, theta, m)
    
    J = (1/(2*m)) * (X*theta - y)' * (X*theta -y);
    
end