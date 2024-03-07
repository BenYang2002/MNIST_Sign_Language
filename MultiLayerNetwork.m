% Multi-Layer Network
% Class that builds a multi-layer
% network, with each layer having a variable number
% of neurons. All neurons have the same transfer function,
% Logsigm. This network uses backpropagation and steepest descent
% as the training algorithim.
classdef MultiLayerNetwork < handle
    properties
        weight_array % Cell array that holds the weight matrix for each layer
        num_of_layers % Number of layers (not including the input layer)
        bias_array % Cell array that holds the bias vector for each layer
        most_recent_outputs % Cell array used in backpropagation, holds the output vector for each layer
        sensitivity_array % Cell array used to get sensitivities of each layer
    end

    methods
        % Constructor. Takes in the number of layers (layer_count) disincluding the input layer
        % and the input/output pair for each layer (layer_details, an array) as arguments and
        % initializes the weight matrix of each layer.
        function obj = MultiLayerNetwork(layer_count, layer_details)
            % Initialize properties from layer_count
            obj.num_of_layers = layer_count;
            obj.most_recent_outputs = cell(1, layer_count + 1); % This Cell array includes the input layer
            obj.weight_array = cell(1, layer_count);
            obj.bias_array = cell(1, layer_count);
            obj.sensitivity_array = cell(1, layer_count); % This Cell array includes the input layer

            % For each neuron layer
            for i = 1:layer_count
                    % Initialize each weight matrix using the respective
                    % input/output pair from layer_details
                    obj.weight_array{i} = 2 * rand(layer_details(2 * i), layer_details(i + (i - 1))) - 1;

                    % Initialize each bias vector using the respective
                    % output value from layer_details
                    obj.bias_array{i} = 2 * rand(layer_details(2* i), 1) - 1;
            end

        end
        
        % Function that computes the output of the network on a passed in
        % input vector. Uses the logsigm transfer function.
        % Saves the output of each layer to be used in backpropagation.
        % Refer to page 11-26 for the full equations.
        function a = forward(obj, input)
            % a0 = p
            result = input';

            % save first output (p)
            obj.most_recent_outputs{1} = result;

            % for each neuron layer
            for i = 1:obj.num_of_layers
                % Compute output
                result = myLogSigmoid(obj.weight_array{i} * result + obj.bias_array{i});

                % Save
                obj.most_recent_outputs{i + 1} = result;
            end

            % Return final result
            a = result;
        end
        
        % Function that updates the weigt matrix and bias vector of each
        % layer using backpropagation. The performance function here is
        % error (target_output - actual_output) squared.
        % Note that the derivative of logsigm is equivalent to (1 - a) (a).
        % Refer to page 11-26 for the full equations.
        function obj = backward(obj, target)

            % create an array of ones to use in the derivative
            one = ones(size(obj.most_recent_outputs{obj.num_of_layers + 1}));

            % Find sensitivity of final layer
            obj.sensitivity_array{obj.num_of_layers} = -2 .* ((one - obj.most_recent_outputs{obj.num_of_layers + 1}) .* ((obj.most_recent_outputs{obj.num_of_layers + 1})) .* (target  - obj.most_recent_outputs{obj.num_of_layers + 1}));

            % Get sensitivity for all other layers
            for i = obj.num_of_layers -1:-1:1
                % create an array of ones to use in the derivative
                one = ones(size(obj.most_recent_outputs{i+1}));
                obj.sensitivity_array{i} = diag((one - obj.most_recent_outputs{i+1}) .* (obj.most_recent_outputs{i+1})) * obj.weight_array{i+1}' * obj.sensitivity_array{i + 1};
            end

            learning_rate = 0.07;

            for i = 1: obj.num_of_layers
                % Update weight matrix for each layer
                obj.weight_array{i} = obj.weight_array{i} - (learning_rate .* (obj.sensitivity_array{i} * obj.most_recent_outputs{i}'));

                % Update bias vector for each layer
                obj.bias_array{i} = obj.bias_array{i} - learning_rate .* obj.sensitivity_array{i};
            end
        end
    end
end