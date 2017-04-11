import numpy as np
import csv

with open('COMS4721_hw4-data/ratings.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
M_train = np.array(rows)
M_train = M_train.astype(np.float)
print(M_train.shape)

with open('COMS4721_hw4-data/ratings_test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	rows = [row for row in readCSV]
M_test = np.array(rows)
M_test = M_test.astype(np.float)
print(M_test.shape)

##### 943 users and 1682 movies, each user has rated at least 20 movies
#### d=10, sigma_sqaure=0.25, lambda=1
N1 = 943
N2 = 1682
sigma_sqaure = 0.25



####training
for times in range(0,10):
	print('times %d'%times)
	u = np.random.multivariate_normal(np.zeros(10),np.identity(10),943)
	v = np.random.multivariate_normal(np.zeros(10),np.identity(10),1682)
	print(u.shape,v.shape)
	for iteration in range(0,100):
		print('iteration %d'%iteration)
		###update u
		for i in range(1,N1+1):
			v_i = v[M_train[M_train[:,0]==i][:,1].astype(np.int)-1,:]
			m_i = M_train[M_train[:,0]==i][:,2]
			sum_mv_i = m_i.dot(v_i)
			sum_vv_i = v_i.T.dot(v_i)
			u[i-1,:] = np.linalg.inv(sigma_sqaure * np.identity(10) + sum_vv_i).dot(sum_mv_i)
		###update v
		for j in range(1,N2+1):
			u_j = u[M_train[M_train[:,1]==j][:,0].astype(np.int)-1,:]
			m_j = M_train[M_train[:,1]==j][:,2]
			sum_mu_j = m_j.dot(u_j)
			sum_uu_j = u_j.T.dot(u_j)
			v[j-1,:] = np.linalg.inv(sigma_sqaure*np.identity(10)+sum_uu_j).dot(sum_mu_j)
		L = 0
		u_omega = u[M_train[:,0].astype(np.int)-1]
		v_omega = v[M_train[:,1].astype(np.int)-1]
		##corresponding u and v in set omega
		sigma_muv = np.sum(M_train[:,2]-np.sum(np.multiply(u_omega,v_omega),axis=1))
		### np.multiply is element-wise product
		print(sigma_muv)
		sigma_u = np.sum(np.multiply(u,u))/2
		print(sigma_u)
		sigma_v = np.sum(np.multiply(v,v))/2
		print(sigma_v)
		L = -sigma_muv - sigma_u - sigma_v
		print('objective %f'%L)





