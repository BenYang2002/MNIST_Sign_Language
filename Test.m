function a = Test(network, data)
    
    result = zeros(size(data,2), 2);

    % For each test value
    for i = 1:size(data, 1)

        % Get test and target inputs
        input = data(i, 2:end);
        input = double(input) / 255;

       result(i, 1) = data(i, 1);

        % Get results
        out = Reformat(forward(network, input));

        for j = 1:size(out)
            if out(j) == 1
                result(i, 2) = j -1;
                break
            end
        end

        
    end
    % Return
    a = result;
end