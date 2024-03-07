training_data = readmatrix('train.csv');
network = MultiLayerNetwork(2, [784, 100, 100, 26]);
test_data = readmatrix('test.csv');
test_data2 = training_data(25001:end,:);
Train(network, training_data);

res = Test(network, test_data);

id = res(:, 1);
label = res(:, 2);

res2 = Test(network, test_data2);
id2 = res2(:, 1);
label2 = res2(:, 2);
expected = test_data2(:,2);
accuracy = calAccuracy(expected,label2);
disp("accuracy");
disp(accuracy);
T = table(id, label);
writetable(T, 'CSS485InterimAttempt.csv');
type CSS485InterimAttempt.csv
