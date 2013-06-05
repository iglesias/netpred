C = [0.01, 0.10, 1.00, 5];

for i = 1:length(C)
    [objective, W1, W2] = sosvm_objective(C(i));
    
    subplot(2,2,i)
    % Make some sample grayscale image
    I = objective / max(objective(:));
    % Convert the grayscale image to RGB
    Irgb = cat(3,I,I,I);
    %Plot the image and some contours in color
    image(W1(1,:), W2(:,1), Irgb)
    axis('xy')
    hold all
    [cplot, h] = contour(W1, W2, I);
    set(h,'ShowText','on')
    text_handle = clabel(cplot, h, 'Color', 'y');
    colormap cool
    xlabel('w_1')
    ylabel('w_2')
    title(sprintf('Regularization C = %.2f', C(i)))
end
