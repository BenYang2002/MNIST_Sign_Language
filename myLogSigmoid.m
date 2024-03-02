% Function that computes the Log Sigmoid [1 / (1 + e ^ -x)]
% function on a vector
function a = myLogSigmoid(x)
    % create return array
    result = zeros(length(x), 1);

    % for each element in passed in vector
    for indx = 1:length(x)
        % get element and make it negative
        new_x = -x(indx);

        % perform Log Sigmoid function on that element
        % and store it in result
        result(indx) = 1 / (1 + exp(new_x));
    end
    % return
    a = result;
end