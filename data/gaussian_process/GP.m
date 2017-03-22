X_test = load('X_test.csv');
X_train = load('X_train.csv');
y_test = load('y_test.csv');
y_train = load('y_train.csv');

[row col] = size(X_train);
Xn = zeros(row, row);
for i = 1:row
    for j = 1:row
        Xn(i,j) = sum((X_train(i,:) - X_train(j,:)).^2);
    end
end

X0 = zeros(42,row);
X = zeros(42,42);
for i = 1:42
    for j = 1:row
        X0(i,j) = sum((X_test(i,:) - X_train(j,:)).^2);
    end 
    for j = 1:42
        X(i,j) = sum((X_test(i,:) - X_test(j,:)).^2);
    end 
end 

b_set = [5,7,9,11,13,15];
sigma2_set = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1];
RMSE = zeros(6,10);
for i = 1:6
    for j = 1:10
        b = b_set(i); sigma2 = sigma2_set(j);
        Kn = exp(-Xn/b); 
        K0 = exp(-X0/b);
        K = exp(-X/b);
        mu = K0 * inv(sigma2 * eye(350) + Kn) * y_train;
        sigma = sigma2 * eye(42) + K - K0 * inv(sigma2 * eye(350) + Kn) * K0';
        RMSE(i,j) = sqrt((y_test - mu)' * (y_test - mu) / 42);
    end     
end



