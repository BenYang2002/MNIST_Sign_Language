function output = softmax(input)
    % Subtract the maximum value for numerical stability
    inputShifted = input - max(input);
    expInput = exp(inputShifted);
    total = sum(expInput);
    output = expInput / total;
end
