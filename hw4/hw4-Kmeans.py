import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

mean1 = [0,0]
mean2 = [3,0]
mean3 = [0,3]
cov =[[1,0],[0,1]]
x1 = np.random.multivariate_normal(mean1,cov,100)
x2 = np.random.multivariate_normal(mean2,cov,250)
x3 = np.random.multivariate_normal(mean3,cov,150)

x = np.concatenate((x1,x2,x3))
print(x.shape)

u = np.random.rand(4,5,2)
c = np.zeros((500,4))

#start from zero, should be modified to start from 1 when plot
print(u)
objective = np.zeros((20,4))
for k in range(2,6):
	for iteration in range(0,20):
		print(iteration)
		for i in range(500):
			l2min = (x[i]-u[k-2,0]).dot((x[i]-u[k-2,0]).T)
			for j in range(1,k):
				tmp = (x[i]-u[k-2,j]).dot((x[i]-u[k-2,j]).T)
				if tmp <l2min:
					l2min = tmp
					c[i,k-2] = j
		nk = np.zeros(k)
		for i in range(0,500):
			nk[c[i,k-2]] += 1
		for i in range(0,k):
			u[k-2,i] = 0
		for i in range(0,500):
			u[k-2,c[i,k-2]] += x[i]/nk[c[i,k-2]]
		for i in range(0,500):
			objective[iteration,k-2] += (x[i]-u[k-2,c[i,k-2]]).dot((x[i]-u[k-2,c[i,k-2]]).T)
		print(objective[iteration,k-2])
print(objective)
rounds = np.linspace(1,21,20)
colorset = ['r','b','g','k']
for k in range(2,6):
	plt.plot(rounds,objective[:,k-2],colorset[k-2],label='k=%d'%k)
plt.xlabel('iteration')
plt.ylabel('objective function')
plt.legend()
plt.show()

colorset = ['r','b','g','y','c']
for k in range(0,3):
	plt.scatter(x[c[:,1]==k][:,0],x[c[:,1]==k][:,1],s=10,color=colorset[k-2],marker='.',label='data k=%d'%k)
plt.scatter(u[1,:3,0],u[1,:3,1],s=50,color='k',marker='*',label='u')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

for k in range(0,5):
	plt.scatter(x[c[:,3]==k][:,0],x[c[:,3]==k][:,1],s=10,color=colorset[k-2],marker='.',label='data k=%d'%k)
plt.scatter(u[3,:,0],u[3,:,1],s=50,color='k',marker='*',label='u')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()







