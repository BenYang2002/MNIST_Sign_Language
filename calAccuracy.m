function accuracy = calAccuracy(ex,pred)
    total = zeros(26,1);
    correct = zeros(26,1);
    accuracy = zeros(26,1);
    for i = 1 : size(ex,1)
        total(ex(i) + 1) = total(ex(i) + 1) + 1;
        if (ex(i) == pred(i))
            correct(ex(i) + 1) = correct(ex(i) + 1) + 1;
        end
    end
    for i = 1 : 26
        accuracy(i) = correct(i) / total(i);
    end
end
