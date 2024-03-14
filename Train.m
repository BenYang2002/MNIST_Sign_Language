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
    totalLY = [];
    prevASE = 10^10;
    prevACE = 10^10;
    while (any(error_matrix(:) > allowable_error))
        if count > 10
            break
        end
        disp("epoch: " + count);
        xplots = [];
        yplots = [];
        ylplots = [];
        mse = 0;
        accumulated_learningRate = 0;
        aLRcounter = 0;
        for i = 1:size(data, 1)
            
            aLRcounter = aLRcounter + 1;
            % Get training vector and corresponding expected output
            input = data(i, 3:end);
            input = double(input) / 255;

            % Convert target from scalar to hot-one encoding
            target = zeros(26,1);
            target(data(i, 2) + 1) = 1;
            

            % Run forward to save results to use in backprop,
            % compute error and save it
            a = forward(network, input);
            if (network.crossEN)
                current_CE = -1 * sum(target .* log(a));
                network.accumulatedCE = network.accumulatedCE + current_CE;
            else
                currentSE = (target - a)' * (target - a);
                network.accumulatedSE = network.accumulatedSE + currentSE;
            end
            if (mod(i,int32(size(data,1)/100)) == 0)
                disp("network.accumulatedSE");
                disp(network.accumulatedSE);
                if (network.crossEN)
                    % ROLL BACK 
                    if (network.accumulatedCE >= prevACE * (1 + network.dynamicTolerance))
                        network.weight_array = network.rollBackW;
                        network.bias_array = network.rollBackB;
                        if (network.currentLR * (1 - network.decreaseRate) > network.lowerBound)
                            network.currentLR = network.currentLR * (1 - network.decreaseRate);
                        else
                            network.currentLR = network.lowerBound;
                        end
                    % keep the result and learning rate
                    elseif (network.accumulatedCE >= prevACE)
                        network.rollBackW = network.weight_array;
                        network.rollBackB = network.bias_array;
                        prevACE = network.accumulatedCE;
                    % increase learning rate
                    else
                        network.rollBackW = network.weight_array;
                        network.rollBackB = network.bias_array;
                        network.currentLR = network.currentLR * (1 + network.increaseRate);
                        prevACE = network.accumulatedCE;
                    end
                    network.accumulatedCE = 0;
                else
                    % ROLL BACK 
                    if (network.accumulatedSE >= prevASE * (1 + network.dynamicTolerance))
                        disp("rolling back");
                        network.weight_array = network.rollBackW;
                        network.bias_array = network.rollBackB;
                        if (network.currentLR * (1 - network.decreaseRate) > network.lowerBound)
                            network.currentLR = network.currentLR * (1 - network.decreaseRate);
                        else
                            network.currentLR = network.lowerBound;
                        end
                    % keep the result and learning rate
                    elseif (network.accumulatedSE >= prevASE)
                        network.rollBackW = network.weight_array;
                        network.rollBackB = network.bias_array;
                        prevASE = network.accumulatedSE;
                    % increase learning rate
                    else
                        network.rollBackW = network.weight_array;
                        network.rollBackB = network.bias_array;
                        if (network.currentLR * (1 + network.increaseRate) > network.upperBound)
                            network.currentLR = network.upperBound;
                        else
                            network.currentLR = network.currentLR * (1 + network.increaseRate);
                        end
                        prevASE = network.accumulatedSE;
                    end
                    network.temp = [network.temp,network.currentLR];
                    network.accumulatedSE = 0;
                    network.accumulatedCE = 0;
                end
            end
            e = target - a;
            pIndex = e'*e;
            mse = mse + pIndex;
            accumulated_learningRate = accumulated_learningRate + network.currentLR;
            if (mod(i,int32(size(data,1)/10)) == 0)
                xplots = [xplots,i];
                yplots = [yplots,mse/i];
                ylplots = [ylplots,accumulated_learningRate/aLRcounter];
                accumulated_learningRate = 0;
                aLRcounter = 0;
                plotting(xplots,yplots,ylplots);
                drawnow();
            end
            error_matrix(:, i) = e;

            % Compute backprop for this specific input
            backward(network, target);
        end
        totalX = [totalX,xplots + ( count - 1 ) * size(data, 1)];
        totalY = [totalY,yplots];
        totalLY = [totalLY,ylplots];
        count = count + 1;
    end
    plotting(totalX,totalY,totalLY);
end
