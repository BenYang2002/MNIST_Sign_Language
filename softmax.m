function output = softmax(input)
    total = sum(exp(input));
    for i = 1 : size(input,1)
        input(i) = exp(input(i)) / total;
    end
    output = input;
end
