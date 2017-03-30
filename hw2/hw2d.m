
Xtrain = csvread('hw2-data/X_train.csv');
ytrain = csvread('hw2-data/y_train.csv');
Xtest = csvread('hw2-data/X_test.csv');
ytest = csvread('hw2-data/y_test.csv');

Xtrain=cat(2,ones(4508,1),Xtrain);
Xtest=cat(2,ones(93,1),Xtest);
w=zeros(58,1);
L=zeros(10000,1);
ytrain(ytrain==0)=-1;
ytest(ytest==0)=-1;

for t = 1:10000
    eta = 1/sqrt(t+1)*(10^-5);
    gradientsum = zeros(58,1);
    Lsum = 0;
    for i = 1:4508
        tmp = ytrain(i)*Xtrain(i,:)*w;
        if tmp<-1.0e80
            sigma = 1/(1+exp(1.0e80));
        elseif tmp>1.0e80
            sigma = 1/(1+exp(-1.0e80));
        else
            sigma = 1/(1+exp(-tmp));
        end
        gradientsum = gradientsum + (1-sigma)*ytrain(i)*Xtrain(i,:)';
        Lsum = Lsum + log(sigma);
    end
    w = w + eta * gradientsum;
    L(t)=Lsum;
end
plot(L)