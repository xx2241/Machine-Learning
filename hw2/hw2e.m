Xtrain = csvread('hw2-data/X_train.csv');
ytrain = csvread('hw2-data/y_train.csv');
Xtest = csvread('hw2-data/X_test.csv');
ytest = csvread('hw2-data/y_test.csv');

Xtrain=cat(2,ones(4508,1),Xtrain);
Xtest=cat(2,ones(93,1),Xtest);
w=zeros(58,1);
L=zeros(100,1);
ytrain(ytrain==0)=-1;
ytest(ytest==0)=-1;
accuracy = zeros(100,1);
for t = 1:100
    eta = 1/sqrt(t+1);
    gradientsum = zeros(58,1);
    gradientsum_2 = zeros(58,58);
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
        tmp2 = Xtrain(i,:)*w;
        if tmp<-1.0e80
            sigma_2 = 1/(1+exp(1.0e80));
        elseif tmp>1.0e80
            sigma_2 = 1/(1+exp(-1.0e80));
        else
            sigma_2 = 1/(1+exp(-tmp));
        end
        gradientsum_2 = gradientsum_2 - sigma_2*(1-sigma_2)*Xtrain(i,:)'*Xtrain(i,:);
        Lsum = Lsum + log(sigma);
    end
    w = w - eta * inv(gradientsum_2)*gradientsum;
    L(t)=Lsum;
    y_pre = zeros(93,1);
    for i = 1:93
        if ((Xtest(i,:)*w>0 & ytest(i)==1) | (Xtest(i,:)*w<0&ytest(i)==-1))
            accuracy(t) = accuracy(t) + 1;
        end
    end
    accuracy(t) = accuracy(t)/93;
end
disp(accuracy(100))
figure(1)
plot(accuracy)
figure(2)
plot(L)