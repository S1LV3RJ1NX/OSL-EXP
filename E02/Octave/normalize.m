function [X_norm, mu, sigma] = normalize(X)
    
    
%   Normalizes the features in X 
%   normalize(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
    
    mu = mean(X); % matrix of means of X values
    sigma = std(X); % matrix of std's of X    
    X_norm = (X - mu) ./ sigma ;

end     