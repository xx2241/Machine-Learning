import numpy as np
import scipy
import csv
from scipy import linalg
from numpy.linalg import inv
import matplotlib.pyplot as plt



with open('hw1-data/X_train.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
x_train = np.array(rows)
x_train = x_train.astype(np.float)
print(x_train)

with open('hw1-data/y_train.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	p = []
	for row in readCSV:
		p.append(row)
y_train = np.array(p)
y_train = y_train.astype(np.float)
print(y_train)

print(x_train.shape)


#################
#######(a)#######
#################

U, s, V = linalg.svd(x_train, full_matrices=False)
S = linalg.diagsvd(s,7,7)
print(U.shape,S.shape,V.shape)
print(V.shape, inv(S).shape,U.T.shape, y_train.shape)

w_ls = V.dot(inv(S)).dot(U.T).dot(y_train)
print(w_ls.shape)


## Although I calculated SVD, I realized that using trace will be more convenient

w_rr = np.zeros((7, 5001))
df = np.zeros(5001)
for i in range(5001):
	w_rr[:,i] = inv(i*np.identity(7) + x_train.T.dot(x_train)).dot(x_train.T).dot(y_train).flatten()
	df[i] = np.trace(x_train.dot(inv(x_train.T.dot(x_train)+i*np.identity(7))).dot(x_train.T))

print(df)
print(w_rr)


plt.plot(df[:],w_rr[0,:], 'r',label='d1')
plt.plot(df[:],w_rr[1,:], 'b',label='d2')
plt.plot(df[:],w_rr[2,:], 'g',label='d3')
plt.plot(df[:],w_rr[3,:], 'y',label='d4')
plt.plot(df[:],w_rr[4,:], 'k',label='d5')
plt.plot(df[:],w_rr[5,:], 'c',label='d6')
plt.plot(df[:],w_rr[6,:], 'm',label='d7')
plt.ylabel('w_rr')
plt.xlabel('df(lambda)')
plt.legend(loc='lower left')
plt.show()

#################
#######(c)#######
#################

with open('hw1-data/X_test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
x_test = np.array(rows)
x_test = x_test.astype(np.float)
print(x_test.shape)

with open('hw1-data/y_test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
y_test = np.array(rows)
y_test = y_test.astype(np.float)
print(y_test.shape)


rmse = np.zeros(51)






for i in range(51):
	rmse[i] = np.sqrt((y_test - x_test.dot(w_rr[:,i]).reshape(42,1)).T.dot(y_test - x_test.dot(w_rr[:,i]).reshape(42,1))/42)

lamb = np.linspace(0,50,51)
print(lamb)

plt.plot(lamb[:],rmse[:], 'r.')
plt.ylabel('RMSE')
plt.xlabel('lambda')
plt.show()

#################
#######(d)#######
#################

tmp2 = np.concatenate((np.square(x_train[:,0:6]),x_train[:,0:6]),axis=1)
x2_train = np.concatenate((tmp2,np.ones((350,1))),axis=1)
tmp3 = np.concatenate((tmp2,np.power(x_train[:,0:6],3)),axis=1)
x3_train = np.concatenate((tmp3,np.ones((350,1))),axis=1)

tmp_test2 = np.concatenate((np.square(x_test[:,0:6]),x_test[:,0:6]),axis=1)
x2_test = np.concatenate((tmp_test2,np.ones((42,1))),axis=1)
tmp_test3 = np.concatenate((tmp_test2,np.power(x_test[:,0:6],3)),axis=1)
x3_test = np.concatenate((tmp_test3,np.ones((42,1))),axis=1)

print(x2_train.shape)
print(x3_train.shape)

w_rr2 = np.zeros((13,501))
for i in range(501):
	w_rr2[:,i] = inv(i*np.identity(13) + x2_train.T.dot(x2_train)).dot(x2_train.T).dot(y_train).flatten()
print(w_rr2)

w_rr3 = np.zeros((19,501))
for i in range(501):
	w_rr3[:,i] = inv(i*np.identity(19) + x3_train.T.dot(x3_train)).dot(x3_train.T).dot(y_train).flatten()
print(w_rr3)
'''
rmse1 = np.zeros(501)

for i in range(501):
	rmse1[i] = np.sqrt((y_test - x_test.dot(w_rr[:,i]).reshape(42,1)).T.dot(y_test - x_test.dot(w_rr[:,i]).reshape(42,1))/42)

rmse2 = np.zeros(501)
for i in range(501):
	rmse2[i] = np.sqrt((y_test - x2_test.dot(w_rr2[:,i]).reshape(42,1)).T.dot(y_test - x2_test.dot(w_rr2[:,i]).reshape(42,1))/42)

rmse3 = np.zeros(501)
for i in range(501):
	rmse3[i] = np.sqrt((y_test - x3_test.dot(w_rr3[:,i]).reshape(42,1)).T.dot(y_test - x3_test.dot(w_rr3[:,i]).reshape(42,1))/42)


lamb_poly = np.linspace(0,500,501)

plt.plot(lamb_poly[:],rmse1[:],'r.',label='p=1')
plt.plot(lamb_poly[:],rmse2[:],'b.',label='p=2')
plt.plot(lamb_poly[:],rmse3[:],'g.',label='p=3')
plt.ylabel('RMSE')
plt.xlabel('lambda')
plt.legend(loc='lower right')
plt.show()
'''
