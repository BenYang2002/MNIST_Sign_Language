training_data = readmatrix('train.csv');
test_data = readmatrix('test.csv');
network = MultiLayerNetwork(2, [784, 30, 30, 26]);

training_data = training_data(1:20000,:);
test_data = training_data(20001:end,:);

Train(network, training_data);

res = Test(network, test_data);

id = res(:, 1);
label = res(:, 2);

T = table(id, label);
writetable(T, 'CSS485InterimAttempt.csv');
type CSS485InterimAttempt.csv
