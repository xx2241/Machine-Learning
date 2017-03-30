import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import scipy.special
from numpy.linalg import pinv

with open('hw2-data/X_train.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
x_train = np.array(rows)
x_train = x_train.astype(np.float)
print(x_train.shape)

with open('hw2-data/y_train.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	p = []
	for row in readCSV:
		p.append(row)
y_train = np.array(p)
y_train = y_train.astype(np.float)
print(y_train.shape)

with open('hw2-data/X_test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
x_test = np.array(rows)
x_test = x_test.astype(np.float)
print(x_test.shape)

with open('hw2-data/y_test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
y_test = np.array(rows)
y_test = y_test.astype(np.float)
print(y_test.shape)

accuracy = np.zeros(20)

dis = np.zeros((93,4508))
for j in range(0,93):
	for k in range(0,4508):
		dis[j,k] += np.sum(np.absolute(x_train[k,:]-x_test[j,:]))
		#dis[j,k] += np.linalg.norm(np.subtract(x_train[k,:],x_test[j,:]),ord=1)
		#for l in range(0,57):
			#dis[j,k] += abs(x_train[k,l] - x_test[j,l])

dis = dis.argsort()
print(dis)


y_knn_pre = np.zeros((20,93))

for k in range(1,21):
	for i in range(0,93):
		sum0 = 0
		sum1 = 0
		for j in range(0,k):
			if(y_train[dis[i,j]]==1):
				sum1 +=1
			else:
				sum0 +=1
		if sum1 > sum0:
			y_knn_pre[k-1,i] = 1
print(y_knn_pre)


knn_accuracy = np.zeros(20)
for k in range(0,20):
	y_correct = 0
	for i in range(0,93):
		if y_test[i]==y_knn_pre[k,i]:
			y_correct += 1
	knn_accuracy[k] = (y_correct/93)
	print(y_correct)
print(knn_accuracy)
plt.plot(knn_accuracy)
plt.show()
