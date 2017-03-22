import numpy as np 
import csv 
import math 
import matplotlib.pyplot as plt

with open('data/boosting/X_train.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
X_train = np.array(rows)
X_train = X_train.astype(np.float)
X_train = np.concatenate((X_train,np.ones((1036,1))),axis=1)
print(X_train.shape)

with open('data/boosting/y_train.csv') as csvfile:
	readCSV = csv.reader(csvfile,delimiter=',')
	p = []
	for row in readCSV:
		p.append(row)
y_train = np.array(p)
y_train = y_train.astype(np.float)
print(y_train.shape)

with open('data/boosting/X_test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
X_test = np.array(rows)
X_test = X_test.astype(np.float)
X_test = np.concatenate((X_test,np.ones((1000,1))),axis=1)
print(X_test.shape)

with open('data/boosting/y_test.csv') as csvfile:
	readCSV = csv.reader(csvfile,delimiter=',')
	p = []
	for row in readCSV:
		p.append(row)
y_test = np.array(p)
y_test = y_test.astype(np.float)
print(y_test.shape)

##initialization
w_bootstrap = np.ones(1036)/1036
print(w_bootstrap)
train_error = np.zeros(1500)
test_error = np.zeros(1500)
y_test_pre = np.zeros(1000)
y_train_pre = np.zeros(1036)
error = 0
alpha = np.ones(1500)
w = np.zeros((1500,6))

for t in range(0,1500):
	print('iteration ' + str(t))
	index_bootstrap = np.random.choice(1036, 1036, p=w_bootstrap)
	X_train_bootstrap = X_train[index_bootstrap,:]
	y_train_bootsrap = y_train[index_bootstrap,:]
	w[t,:] = np.squeeze(np.linalg.inv(X_train_bootstrap.T.dot(X_train_bootstrap)).dot(X_train_bootstrap.T).dot(y_train_bootsrap))
	train_committee = np.zeros(1036)
	test_committee = np.zeros(1000)

	y_train_pre = X_train.dot(w[t,:].reshape(6,1))
	for i in range(0,1036):
		if(y_train_pre[i]>0):
			y_train_pre[i] = 1
		else:
			y_train_pre[i] = -1
		if(y_train_pre[i]!=y_train[i,0]):
			error = error + w_bootstrap[i]
	while error > 0.5:
		error = 0
		print('flipping')
		w[t,:] = -w[t,:]
		y_train_pre = X_train.dot(w[t,:].reshape(6,1))	
		for i in range(0,1036):
			if(y_train_pre[i]>0):
				y_train_pre[i] = 1
			else:
				y_train_pre[i] = -1
			if(y_train_pre[i]!=y_train[i,0]):
				error = error + w_bootstrap[i]
	if error < math.pow(10,-300):
		error = math.pow(10,-300)
	print(error)
	alpha[t] = math.log((1-error)/error)/2
	error = 0
	sigma = 0

	for k in range(0,t+1):
		y_train_pre = X_train.dot(w[k,:].reshape(6,1))
		y_test_pre = X_test.dot(w[k,:].reshape(6,1))
		for i in range(0,1036):
			if(y_train_pre[i]>0):
				y_train_pre[i] = 1
			else:
				y_train_pre[i] = -1
		for i in range(0,1000):
			if(y_test_pre[i]>0):
				y_test_pre[i] = 1
			else:
				y_test_pre[i] = -1
		train_committee = train_committee + alpha[k] * np.squeeze(y_train_pre)
		test_committee = test_committee + alpha[k] * np.squeeze(y_test_pre)

	for i in range(0,1036):
		if((train_committee[i]>0 and y_train[i,0]==-1) or (train_committee[i]<0 and y_train[i,0]==1)):
			train_error[t] = train_error[t]+1
	for i in range(0,1000):
		if((test_committee[i]>0 and y_test[i,0]==-1) or (train_committee[i]<0 and y_train[i,0]==1)):
			test_error[t] = test_error[t]+1
	print(train_error[t])
	print(test_error[t])

	for i in range(0,1036):
		w_bootstrap[i] = w_bootstrap[i]*math.exp(-alpha[t]*y_train[i,0]*y_train_pre[i])
		sigma = sigma + w_bootstrap[i]
	for i in range(0,1036):
		w_bootstrap[i] = w_bootstrap[i]/sigma


train_error = train_error/1036
test_error = test_error/1000

rounds = np.linspace(0,1500,1500)

plt.plot(rounds[:],train_error[:], 'r.', label='training error')
plt.plot(rounds[:],test_error[:], 'b.', label='testing error')
plt.ylabel('error')
plt.xlabel('iteration')
plt.show()


