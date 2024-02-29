    classdef BackPropLayer < handle
    %BACKPROPLAYER Summary of this class goes here
    %   Detailed explanation goes here
    properties
        % Network Parameters
        layers          % Cell array containing the arguments for each layer
        transfer
        learning_rate
        mini_batchUp    % Boolean that activates the mini-batch update
        mini_batchSize  % Size of mini-batch
        MNIST           % Boolean to modify output from a vector to a real number
        
        % Training Parameters
        training            % Boolean indicating training or prediction mode
        trainingTimes       % Maximum number of times going through the training set
        trainingSize        % Size of the training set
        plottingEpoch = 0   % Epoch for plotting
        
        % Results 
        aLayers            % Cell array containing the output of each layer
        nLayers            % Cell array containing the net input of each layer
        prediction
        mse                % Mean squared error
        sensitivity_Matrix % Numeric matrix, each column represents sensitivity
        
        % Plotting Coordinates
        xplots          % Array of x coordinates for plotting
        yplots          % Array of y coordinates for plotting
    end

    methods
        function this = BackPropLayer(weightRow, weightColumn, ...
                              learning_rate, transfer, training, ...
                              trainingTimes, MNIST, miniBatchup, batchSize)
            % BACKPROPLAYER Construct an instance of this class
            
            % Check if the dimensions of weightRow and weightColumn match
            if size(weightRow, 2) ~= size(weightColumn, 2)
                error("Dimension of weightRow and weightColumn " + ...
                      "doesn't match");
            end
            
            % Initialize layers with random weight matrices and bias 
            % vectors
            for i = 1:size(weightColumn, 2)
                weightMatrix = rand(weightRow(i), weightColumn(i)) * 0.1;
                biasVec = rand(weightRow(i), 1);
                this.layers{i} = [weightMatrix, biasVec];
            end
            
            % Set other properties
            this.learning_rate = learning_rate;
            this.transfer = transfer;
            this.training = training;
            this.trainingTimes = trainingTimes;
            this.MNIST = MNIST;
            this.mini_batchUp = miniBatchup;
            this.mini_batchSize = batchSize
        end

        function [output] = forward(this, input)
            % FORWARD
            
            % Input is a vector from the previous layer or from the input layer
            this.aLayers{1} = input;
            
            % Iterate through layers
            for i = 1:size(this.layers, 2)
                parameterM = this.layers{i};
                
                % Compute net input for the layer
                layerNetInput = parameterM(:, 1:end-1) * input + parameterM(:, end);
                this.nLayers{i} = layerNetInput; % Save net input
                
                % Compute layer output using activation function
                layerOut = this.activationFunc(layerNetInput, this.transfer{i});
                input = layerOut; % Update feed vector
                this.aLayers{i+1} = layerOut; % Save layer output
            end
            
            % Get final output
            this.prediction = this.aLayers{end};
            
            % Modify output if needed
            if (~this.training)
                 this.prediction = this.modifyOutput(this.prediction);
                this.aLayers{end} = this.prediction;
            end
        end

        function output = modifyOutput(this, input)
            % Find element with largest confidence and use it as prediction
            % input is the raw output of the network

            % Find the maximum value and its index
            max = 0;
            out = 0;
            for i = 1:size(input, 1)
                if max < input(i)
                    max = input(i);
                    out = i;
                end
            end
            
            % Modify output based on the MNIST flag
            if this.MNIST
                output = out - 1;
                return;
            end
            
            % Convert output to a one-hot encoded vector
            output = zeros(size(input, 1), 1);
            output(out) = 1;
        end

        function der = takeDeravative(this, funcName, input)
            % TAKEDERaVATIVE
            % funcName specifies the activation function name
            % Input is the net input of m layer
            
            % Check which activation function to use
            if isequal(funcName, "sigmoid")
                % Compute derivative for sigmoid activation function
                result = this.sigmoid(input);
                d = zeros(size(result, 1), 1);
                for i = 1:size(result, 1)
                    d(i) = result(i) * (1 - result(i));
                end
                der = diag(d);
                return;
            end
            
            if isequal(funcName, "softmax")
                % Compute derivative for softmax activation function
                sumS = sum(exp(input));
                denominator = sumS^2;
                der = zeros(size(input, 1), size(input, 1));
                for column = 1:size(input, 1)
                    for row = 1:size(input, 1)
                        if row == column
                            der(row, column) = (exp(input(row)) * sumS - exp(input(row))^2) / denominator;
                        else
                            der(row, column) = -1 * (exp(input(row)) * exp(input(column))) / denominator;  
                        end
                    end
                end
                return;
            end
            
            % If no matching transfer function is found
            der = -1;
            error("No matching transfer function");
        end

        function train(this, inputMatrix, expectedM)
            % sample training 
            % TODO : Decide whether to update the terminating condition
            % inputMatrix assume that each column is an input
            % expectedM each column is one corresponding expected output
            epoch = 1 ;
            correct = false;
            this.trainingSize = size(inputMatrix,2);
            while (~correct && epoch <= this.trainingTimes)
                correct = true;                   
                %disp("iter: " + epoch);
                iter = 1;
                this.mse = 0;
                if ( this.mini_batchUp )
                    disp("epoch " + epoch);
                    correct = false;
                    remaining = 0;
                    for i = 1 : ( size(inputMatrix,2) / ...
                            this.mini_batchSize )
                        start = (i-1) * this.mini_batchSize + 1;
                        endIndex = i * this.mini_batchSize;
                        expectedOut = expectedM(:, start : endIndex );
                        remaining = endIndex + 1;
                        predictions = zeros(10, this.mini_batchSize);
                        A = cell(1,size(this.layers,2)+1);  
                        % A is a 1D array
                        % A{i} holds the output matrix for ith element
                        N = cell(1,size(this.layers,2));
                        % N{i} holds the netinput matrix for ith layer
                        % N{i,j} holds the netinput for ith layer and 
                        % jth element
                        disp("start " + start);
                        disp("endIndex " + endIndex);
                        for i = start : endIndex
                            input = inputMatrix(:,i);
                            predictions(:,i - start + 1) = this.forward(input);
                            for n = 1 : size(this.nLayers,2)
                                N{n} = [N{n}(:,1:end),this.nLayers{n}];
                            end
                            for j = 1 : size(this.aLayers,2)
                                A{j} = [A{j}(:,1:end),this.aLayers{j}];
                            end
                        end
                        disp(predictions);
                        miniBatchUpdate(this,expectedOut,predictions,A,N);
                    end
                    if (remaining <= this.trainingSize) 
                        start = remaining;
                        if (start == 0)
                            start = 1; % handle the case when total 
                            % training size is smaller than the batch
                        end
                        endIndex = size(inputMatrix,2);
                        expectedOut = inputMatrix(:,start : endIndex);
                        predictions = zeros(10, endIndex - start + 1);
                        A = cell(1,size(this.layers,2)+1);
                        N = cell(1,size(this.layers,2));
                        for i = start : endIndex
                            input = inputMatrix(:,i);
                            predictions(:,i - start + 1) = this.forward(input);
                            for n = 1 : size(this.nLayers,2)
                                N{n} = [N{n}(:,1:end),this.nLayers{n}];
                            end
                            for j = 1 : size(this.aLayers,2)
                                A{j} = [A{j}(:,1:end),this.aLayers{j}];
                            end
                        end
                        miniBatchUpdate(this,expectedOut,predictions,A,N);
                    end
                else
                    for i = 1 : size(inputMatrix,2)
                        disp("iter: " + iter);
                        iter = iter + 1;
                        input = inputMatrix(:,i);
                        ex = expectedM(:,i);
                        this.forward(input);
                        this.backwardUpdate(ex);
                        ex = outputToVec(this,ex);
                        pIndex = (ex - this.prediction)' * ...
                            (ex - this.prediction);
                        if (mod(iter,int32(this.trainingSize / 100)) == 0)
                            plotting(this,ex,iter,epoch);
                        end
                        if ~isequal(this.prediction,expectedM(:,i))
                            this.mse = this.mse + pIndex(1,1);
                            mseIter = pIndex;
                            disp("ex is " + ex);
                            disp("prediction is " + this.prediction);
                            disp("average mse over " + iter + " iter is " ...
                                + (this.mse / iter));
                            disp("mse of iter: " + iter + " is " + mseIter);
                            disp("error:");
                            disp("predicion is");
                            disp(this.prediction);
                            disp("expected output is");
                            disp(expectedM(:,i));
                            correct = false;
                        end
                    end
                end
                epoch = epoch + 1;
            end
            title('Performance Index Over i Iterations');
            xlabel('Number of Iterations');
            ylabel('Performance Index');
            hold off
        end

        function plotting(this,ex,iter,epoch)
            if (this.plottingEpoch ~= epoch)
                this.plottingEpoch = epoch;
                this.xplots = [];
                this.yplots = [];
                hold off;
            end
            xlim([iter - ( this.trainingSize / 10 ), ...
                iter + ( this.trainingSize / 10 )]); 
            % Set x-axis limits from 0 to 10
            ylim([0, 1.5]); % Set y-axis limits from -1 to 1
            pIndex = (ex - this.prediction)' * (ex - this.prediction);
            disp("expected " + ex);
            disp("prediction " + this.prediction);
            disp("Performance index: " + pIndex);
            this.xplots = [this.xplots,iter];
            this.yplots = [this.yplots,this.mse / iter];
            plot(this.xplots, this.yplots, 'ko-');
            drawnow();
            hold on
        end

        function output = outputToVec(this,expectedOut)
            exOutMod = zeros(size(this.aLayers{end},1),1);
            exOutMod(expectedOut+1) = 1; 
            output = exOutMod; % we map the output from a scalar 
            % to the vector
        end

        function miniBatchUpdate(this,expectedOut,predictions,A,N)
             if (this.MNIST)
                 batchSize = size(predictions,2);
                 S = cell(1,size(this.layers,2)); 
                 % holds the sensitivity matrix for all
                 % layers and for all element in the batch
                 FMmatrix = cell(1,size(predictions,2));
                 TminuxA = zeros(size(predictions,1),size(predictions,2));
                 % holds the matrix of t -a 
                 temp = zeros(10,size(expectedOut,2));
                 for i = 1 : size(expectedOut,2)
                     temp(:,i) = outputToVec(this,expectedOut(i));
                 end
                 expectedOut = temp;
                 TminuxA = expectedOut - predictions;
                 % now proceed to calculate deravative
                 for i = 1 : size(expectedOut,2) % LOOP THROUGH EACH ELE
                     errorOut = TminuxA(:,i);
                     netV = N{end}(:,i);
                     der = this.takeDeravative(this.transfer{end},netV);
                     sM = -2 * der * errorOut; % calculated the sensitivity for
                     % the last layer
                     S{end} = [S{end},sM];
                     prevSense = sM;
                     % calculate all sensitivity
                     for j = size(this.layers,2) : -1 : 2
                        netV = N{j-1}(:,i);
                        der = this.takeDeravative(this.transfer{j-1},netV);
                        sCurrent = der * this.layers{j}(:,1:end-1)' * prevSense;
                        % sCurrent is the sensitivity of the current layer
                        prevSense = sCurrent; 
                        S{j-1} = [S{j-1},sCurrent];
                     end
                 end
                 % S the sensitivity cell
                 % S{i} holds the sensitivity matrix for ith element
                 % S{i}(:,j) holds the sensitivity vec for the jth
                 % layer
                 % start to update
                 for i = 1 : size(this.layers,2)
                     decre = S{i} * A{i}';
                     weights = this.layers{i}(:,1:end-1);
                     bias = this.layers{i}(:,end);
                     weights = weights - ...
                        (this.learning_rate / batchSize) * decre;
                     sS = zeros(size(S{i},1),1);
                     for j = 1 : size(S{i},2)
                        sS = sS + S{i}(:,j);
                     end
                     bias = bias - (this.learning_rate / batchSize) * sS;
                     this.layers{i} = [weights,bias];
                 end
             end
             
        end

        function backwardUpdate(this,expectedOut)
             %%Compare # of neurons to size of error vector
             % This is the function that updates the weight_matrix based 
             % on a single input
             if (this.MNIST)
                 expectedOut = outputToVec(this,expectedOut);
             end
             errorOut = expectedOut - cell2mat(this.aLayers(:,end));
             netV = cell2mat(this.nLayers(:,end));
             der = this.takeDeravative(this.transfer{end},netV);
             sM = -2 * der * errorOut; % calculated the sensitivity for
             % the last layer
             this.sensitivity_Matrix{size(this.layers,2)} = [sM];
             prevSense = this.sensitivity_Matrix{end};
             % calculate all sensitivity
             for i = size(this.layers,2) : -1 : 2
                netV = cell2mat(this.nLayers(:,i-1));
                der = this.takeDeravative(this.transfer{i-1},netV);
                %disp(size(this.layers{i}(:,1:end-1)));
                %disp(size(prevSense));
                sCurrent = der * this.layers{i}(:,1:end-1)' * prevSense;
                % sCurrent is the sensitivity of the current layer
                prevSense = sCurrent; 
                this.sensitivity_Matrix{i-1} = sCurrent;
             end
             % now we have the sensitivity matrix 
             % update weight matrix and bias
             for i = 1 : size(this.layers,2) 
                wM = this.layers{i}(:,1:end - 1); % weight matrix
                b = this.layers{i}(:,end); % bias
                s = this.sensitivity_Matrix{i};
                prevA = (this.aLayers{i})';
                wM = wM - this.learning_rate * s * prevA;
                b = b - this.learning_rate * s;
                this.layers{i} = [wM,b];
             end
        end

        function output = activationFunc(this,input,funcName)
           if (funcName == "sigmoid")
               output = this.sigmoid(input);
               return;
           end
           if (funcName == "softmax")
               output = this.softmax(input);
               return;
           end
           output = input;
        end

        function output = sigmoid(this,input)
            output = input;
            for i = 1 : size(input,1)
                output(i) = 1 / (1 + exp(1)^(-input(i)));
            end
        end

        function output = softmax(this,input)
            total = sum(exp(input));
            for i = 1 : size(input,1)
                input(i) = exp(input(i)) / total;
            end
            output = input;
        end
        
        function print(this)
            for i = 1 : size(this.layers,2)     
                disp("Weight Matrix for layer " + i);
                disp(this.layers{i}(:,1:end-1));
                disp("Bias Vector for layer " + i);
                disp(this.layers{i}(:,end));
            end
        end
    end
end
