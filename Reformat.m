% Function that reformats the output of the network
% to resemble the expected output.
% Sets the element with the largest value to 1
% and the rest of the elements to 0.
function a = Reformat(in)
    % Initial result array
    result = zeros(size(in));

    % Used to find largest element
    largest_element = 0;

    % Used to keep track of largest element
    element_index = 1;

    % For each element in the input
    for i = 1:length(in)
        % The largest element so far
        if in(i) > largest_element

            % Store index and value of that element
            largest_element = in(i);
            element_index = i;
        end
    end

    % Set the index of the largest element to 1 and return
    result(element_index) = 1;
    a = result;
end