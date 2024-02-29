% Read the CSV file into a table
%T = readtable('train.csv');
%labels = T.label;
%data = [];
%for i = 3 : size(T,2)
%    data = [data,table2array(T(2:size(T,1),i))];
%end
%data = data';
% Save the matrix to a MAT-file
%save('training_data.mat', 'data');
%save('training_labels.mat', 'labels');
load('training_data.mat');
load('training_labels.mat');
dataSize = size(data,2);
trainData = data(:,1:(dataSize/2));
trainLabel = labels(:,1:(dataSize/2));
testData = data(:,(dataSize/2)+1 : end);
testLabel = labels(:,(dataSize/2)+1 : end);

trainingTimes = 10;
learning_rate = 0.02;
istraining = true;
isMNIST = true;
weightRows = [256,25];
weightColumns = [784,256];
transferFunc = {"sigmoid","softmax"};
obj_MNIST = BackPropLayer(weightRows,weightColumns,learning_rate,transferFunc, ...
    istraining,trainingTimes,isMNIST);
obj_MNIST.train(trainData,trainLabel);

%%%%%%%%%%%%%%%%%%%%%%%training%%%%%%%%%%%%%%%%%%%%
%T = readtable('test.csv');
%test_labels = T.label;
%test_labels = test_labels';
%test_data = [];
%for i = 3 : size(T,2)
%    test_data = [data,table2array(T(2:size(T,1),i))];
%end
%test_data = test_data';
% Save the matrix to a MAT-file
%save('test_training_data.mat', 'test_data');
%save('test_training_labels.mat', 'test_labels');

obj_MNIST.training = false;
correctCount = zeros(1,25);
totalCount = zeros(1,25);
for i = 1 : size(testData,2)
    input = testData(:,i);
    ex = testLabel(:,i);
    obj_MNIST.forward(input);
    totalCount(ex+1) = totalCount(ex+1) + 1;
    if (isequal(ex,obj_MNIST.prediction))
        correctCount(ex+1) = correctCount(ex+1) + 1;
    end
end
accuracy = zeros(1,10);
for i = 1 : size(correctCount,2)
    accuracy(i) = correctCount(i) / totalCount(i);
end
disp(accuracy);
