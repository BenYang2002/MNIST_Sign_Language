function Train(network, data)
% allowable maximum for error
    allowable_error = 0.2;
    % tracks the error for each term of each input
    error_matrix = ones(26, size(data, 2));
    % only stop training once the error on each term of the input patterns has
    % reached acceptable range
    count = 0;
    while (any(error_matrix(:) > allowable_error))
        if count > 100
            break
        end
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
            disp("targer " + size(target));
            disp("a " + size(a));
            e = target - a;
            disp("targer " + target);
            disp("a " + a);
            pIndex = e'*e;
            mse = mse + pIndex;
            if (mod(i,size(data,1)/100) == 0)
                xplots = [xplots,i];
                yplots = [yplots,mse/i];
                if (mod(i,int32(size(data,1)/5)) == 0)
                    xplots = [];
                    yplots = [];
                end
                plotting(xplots,yplots);
                drawnow();
            end
            error_matrix(:, i) = e;

            % Compute backprop for this specific input
            backward(network, target);
        end

        count = count + 1;
    end
    disp(count)
end
