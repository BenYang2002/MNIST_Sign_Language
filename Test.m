function Train(network, data)
% allowable maximum for error
    allowable_error = 0.2;
    % tracks the error for each term of each input
    error_matrix = ones(26, size(data, 2));
    % only stop training once the error on each term of the input patterns has
    % reached acceptable range
    count = 1;
    totalX = [];
    totalY= [];
    while (any(error_matrix(:) > allowable_error))
        if count > 10
            break
        end
        disp("epoch: " + count);
        xplots = [];
        yplots = [];
        mse = 0;
        for i = 1:size(data, 1)

            % Get training vector and corresponding expected output
            input = data(i, 3:end);
            input = double(input) / 255;

            % Convert target from scalar to hot-one encoding
            target = zeros(26,1);
            target(data(i, 2) + 1) = 1;
            

            % Run forward to save results to use in backprop,
            % compute error and save it
            a = forward(network, input);
            e = target - a;
            pIndex = e'*e;
            mse = mse + pIndex;
            if (mod(i,int32(size(data,1)/10)) == 0)
                xplots = [xplots,i];
                yplots = [yplots,mse/i];
                plotting(xplots,yplots);
                drawnow();
            end
            error_matrix(:, i) = e;

            % Compute backprop for this specific input
            backward(network, target);
        end
        totalX = [totalX,xplots + ( count - 1 ) * size(data, 1)];
        totalY = [totalY,yplots];
        count = count + 1;
    end
    plotting(totalX,totalY);
end
