function [objective, W1, W2] = sosvm_objective(C)
%
%   sosvm_objective(C)
%
%           C           regularizer
%

% \mathcal{X} = \mathbb{R}
% \mathcal{Y} = {-1, +1}
Yi = [-1, +1];

%%% training set

% features
X = [-10, -4, 6, 5];
% labels
Y = [+1, +1, -1, -1];
% number of training examples
assert(length(X) == length(Y))
N = length(X);

w1 = linspace(-3, 5, 1e2);
w2 = linspace(-2, 3, 1e2);
[W1, W2] = meshgrid(w1, w2);

sum = zeros(size(W1));
loss = zeros([size(W1), length(Yi)]);

%%% compute objective

for n = 1:N
    for i = 1:length(Yi)
        y = Yi(i);
        
        % Structured loss contribution
        Delta = y ~= Y(n);
        
        % Joint feature map
        PsiPred = joint_map(X(n), y);
        
        % Dot product with the weight vector
        loss(:,:,i) = Delta + W1*PsiPred(1) + W2*PsiPred(2);
    end
    
    % Joint feature map with ground truth
    PsiTruth = joint_map(X(n), Y(n));
    
    sum = sum + ...
        max(loss(:,:,1), loss(:,:,2)) - (W1*PsiTruth(1) + W2*PsiTruth(2));
end

objective = 1/2*(W1.^2 + W2.^2) + C/N*sum;

end

function Psi = joint_map(x,y)
    if y == +1
        Psi = [x, 0];
    elseif y == -1
        Psi = [0, x];
    else
        error('Elements in Y can only be either +1 or -1')
    end
end
