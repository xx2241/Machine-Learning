clc;
clear;

% data format: user_id movie_id rating
data = load('ratings.csv');
test = load('ratings_test.csv');
sigma2 = 0.25;
d = 10;
lambda = 1;
T = 100;

% analyse the data
maxNum = max(data);
userNum = maxNum(1);
movieNum = maxNum(2);
% user i rated movie j
rated = zeros(userNum, movieNum);
ratingMatrix = zeros(userNum, movieNum);
for i = 1:95000
    rated(data(i,1),data(i,2)) = 1;
    ratingMatrix(data(i,1),data(i,2)) = data(i,3);
end

obj_value = -1 * 10^5;
objTotal = zeros(10,100);
table = zeros(10,2);

for n = 1:10
    % generate v
    v = mvnrnd(zeros(1,d), lambda*eye(d), movieNum);
    u = zeros(userNum, d);
    RMSE = zeros(T);
    % MAP calculation
    for t = 1:T
        % update user position
        for i = 1:userNum
            movie = find(rated(i,:) == 0);
            user = find(data(:,1) == i);
            vj = v;
            vj(movie,:) = 0;
            % calculate the last part
            [m, z] = size(user);
            total = zeros(1,d);
            for j = 1: m
                total = total + data(user(j),3) * v(data(user(j),2),:);
            end
            ui = pinv(lambda*sigma2*eye(d) + vj'*vj) * total';
            u(i,:) = ui';
        end

        % update object position
        for i = 1:movieNum
            user = find(rated(:,i) == 0);
            movie = find(data(:,2) == i);

            ui = u;
            ui(user,:) = 0;
            % calculate the last part
            [m, z] = size(movie);
            total = zeros(1,d);
            for j = 1: m
                total = total + data(movie(j),3) * u(data(movie(j),1),:);
            end
            vj = pinv(lambda*sigma2*eye(d) + ui'*ui) * total';
            v(i,:) = vj';
        end

        result = u * v';
        % calculate objective function
        obj = 0.5 / sigma2 * sum(sum((ratingMatrix-result.*rated).^2));
        obj = obj + 0.5 * lambda * trace(u * u') + 0.5 * lambda * trace(v * v');

        objTotal(n,t) = -obj;
        RMSE(t) = 0;
        for i = 1:5000
            RMSE(t) = RMSE(t) + (test(i,3)-result(test(i,1),test(i,2)))^2;
        end
        RMSE(t) = sqrt(RMSE(t)/5000);
    end
    
    % calculate RMSE

    
    % save the result
    table(n,1) = RMSE;
    table(n,2) = -obj;
    
    % find the best result
    if -obj > obj_value 
        final_v = v;
        final_u = u;
        obj_value = -obj;
    end
end

% plot picture
for i = 1:10
    plot(2:T, objTotal(i,2:T));
    hold on;
end

% find the minimum distance 10 to selected movies
starWar = 50;
myFairLady = 485;
Goodfellas = 182;

nearest = zeros(3,10);
nearest_distance = zeros(3,10);
v = final_v;
% starWar
distance = (v - ones(movieNum,1) * v(50,:)) .* (v - ones(movieNum,1) * v(50,:));
distance = sum(distance');
distance = sqrt(distance);
sorting = sort(distance);
for i = 1:10
    nearest(1,i) = find(distance == sorting(i+1));
    nearest_distance(1,i) = distance(nearest(1,i));
end

% myFairLady
distance = (v - ones(movieNum,1) * v(485,:)) .* (v - ones(movieNum,1) * v(485,:));
distance = sum(distance');
distance = sqrt(distance);
sorting = sort(distance);
for i = 1:10
    nearest(2,i) = find(distance == sorting(i+1));
    nearest_distance(2,i) = distance(nearest(2,i));
end

% Goodfellas
distance = (v - ones(movieNum,1) * v(182,:)) .* (v - ones(movieNum,1) * v(182,:));
distance = sum(distance');
distance = sqrt(distance);
sorting = sort(distance);
for i = 1:10
    nearest(3,i) = find(distance == sorting(i+1));
    nearest_distance(3,i) = distance(nearest(3,i));
end

