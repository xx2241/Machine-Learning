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

#######(a)

n = y_train.shape[0]

pi = np.sum(y_train)/n

count0 = 0
thetay1_0 = np.zeros(54)
for i in range(0,4508):
	if(y_train[i]==0):
		thetay1_0 += x_train[i,0:54] 
		count0 += 1
thetay1_0 /= count0

count1 = 0
thetay1_1 = np.zeros(54)
for i in range(0,4508):
	if(y_train[i]==1):
		thetay1_1 += x_train[i,0:54]
		count1 += 1
thetay1_1 /= count1



tmp54 = 0
tmp55 = 0
tmp56 = 0
for i in range(0,4508):
	if(y_train[i]==0):
		tmp54 += np.log(x_train[i,54])
		tmp55 += np.log(x_train[i,55])
		tmp56 += np.log(x_train[i,56])
thetay2_0 = np.array([count0/tmp54,count0/tmp55,count0/tmp56])

tmp54 = 0
tmp55 = 0
tmp56 = 0
for i in range(0,4508):
	if(y_train[i]==1):
		tmp54 += np.log(x_train[i,54])
		tmp55 += np.log(x_train[i,55])
		tmp56 += np.log(x_train[i,56])
thetay2_1 = np.array([count1/tmp54,count1/tmp55,count1/tmp56])



print(x_train[:,0:54],x_train[:,54:57],thetay2_0,thetay2_1,thetay1_0,thetay1_1)




y_nb_pre = np.zeros(93)
for i in range(0,93):
	tmp0 = 1-pi
	tmp1 = pi
	for j in range(0,54):
		tmp0 *= math.pow(thetay1_0[j],x_test[i,j])*math.pow(1-thetay1_0[j],1-x_test[i,j])
		tmp1 *= math.pow(thetay1_1[j],x_test[i,j])*math.pow(1-thetay1_1[j],1-x_test[i,j])
	for j in range(54,57):
		tmp0 *= thetay2_0[j-54] * math.pow(x_test[i,j],-thetay2_0[j-54]-1)
		tmp1 *= thetay2_1[j-54] * math.pow(x_test[i,j],-thetay2_1[j-54]-1)
	if(tmp1>tmp0):
		y_nb_pre[i] = 1
	else:
		y_nb_pre[i] = 0

print(y_nb_pre)



y_00 = 0
y_01 = 0
y_10 = 0
y_11 = 0
for i in range(0,93):
	if y_test[i]==0 and y_nb_pre[i]==0 :
		y_00 += 1
	elif y_test[i]==0 and y_nb_pre[i]==1:
		y_01 += 1
	elif y_test[i]==1 and y_nb_pre[i]==0:
		y_10 +=1
	else:
		y_11 +=1
print(y_00,y_01,y_10,y_11)






markerline,stemlines, baseline = plt.stem(np.linspace(1,54,54),thetay1_0, '-.')
plt.setp(baseline, 'color', 'r', 'linewidth', 2)
plt.show()
markerline,stemlines, baseline = plt.stem(np.linspace(1,54,54),thetay1_1, '-.')
plt.setp(baseline, 'color', 'b', 'linewidth', 2)

plt.show()