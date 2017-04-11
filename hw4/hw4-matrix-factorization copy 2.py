import numpy as np
import csv
import matplotlib.pyplot as plt

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
#### Star Wars 49, GoodFellas 181, My Fair Lady 484
N1 = 943
N2 = 1682
sigma_sqaure = 0.25
rmse = np.zeros((10,100))
L = np.zeros((10,100))
u = np.zeros((10,943,10))
v = np.zeros((10,1682,10))
movie_distance = np.zeros((10,1682))
Star_Wars_10index = np.zeros((10,10))
Star_Wars_10movie = np.zeros((10,10))
Star_Wars_10distance = np.zeros((10,10))
GoodFellas_10index = np.zeros((10,10))
GoodFellas_10movie = np.zeros((10,10))
GoodFellas_10distance = np.zeros((10,10))
My_Fair_Lady_10index = np.zeros((10,10))
My_Fair_Lady_10movie = np.zeros((10,10))
My_Fair_Lady_10distance = np.zeros((10,10))
for i in range(0,10):
	u[i] = np.random.multivariate_normal(np.zeros(10),np.identity(10),943)
	v[i] = np.random.multivariate_normal(np.zeros(10),np.identity(10),1682)
####training
for times in range(0,10):
	print('times %d'%times)
	for iteration in range(0,100):
		print('iteration %d'%iteration)
		###update u
		for i in range(1,N1+1):
			v_i = v[times][M_train[M_train[:,0]==i][:,1].astype(np.int)-1,:]
			m_i = M_train[M_train[:,0]==i][:,2]
			sum_mv_i = m_i.dot(v_i)
			sum_vv_i = v_i.T.dot(v_i)
			u[times,i-1,:] = np.linalg.inv(sigma_sqaure * np.identity(10) + sum_vv_i).dot(sum_mv_i)
		###update v
		for j in range(1,N2+1):
			u_j = u[times][M_train[M_train[:,1]==j][:,0].astype(np.int)-1,:]
			m_j = M_train[M_train[:,1]==j][:,2]
			sum_mu_j = m_j.dot(u_j)
			sum_uu_j = u_j.T.dot(u_j)
			v[times,j-1,:] = np.linalg.inv(sigma_sqaure*np.identity(10)+sum_uu_j).dot(sum_mu_j)
		u_omega = u[times][M_train[:,0].astype(np.int)-1]
		v_omega = v[times][M_train[:,1].astype(np.int)-1]
		##corresponding u and v in set omega
		sigma_muv = np.sum((M_train[:,2]-np.sum(np.multiply(u_omega,v_omega),axis=1))**2)
		### np.multiply is element-wise product
		print(sigma_muv)
		sigma_u = np.sum(np.multiply(u[times],u[times]))/2
		print(sigma_u)
		sigma_v = np.sum(np.multiply(v[times],v[times]))/2
		print(sigma_v)
		L[times,iteration] = -sigma_muv - sigma_u - sigma_v
		print('objective %f'%L[times,iteration])
		u_test_omega = u[times][M_test[:,0].astype(np.int)-1]
		v_test_omega = v[times][M_test[:,1].astype(np.int)-1]
		sigma_muv_test = np.sum((M_test[:,2] - np.sum(np.multiply(u_test_omega,v_test_omega),axis=1))**2)
		rmse[times,iteration] = np.sqrt(sigma_muv_test/5000)
		print('rmse %f'%rmse[times,iteration])


	Star_Wars = v[times,49,:]
	GoodFellas = v[times,181,:]
	My_Fair_Lady = v[times,484,:]

	for j in range(0,N2):
		movie_distance[times,j] = np.sqrt(np.sum((v[times,j,:] - Star_Wars)**2))
	Star_Wars_10index[times] = np.argsort(movie_distance[times])[1:11]
	Star_Wars_10distance[times] = np.sort(movie_distance[times])[1:11]
	print(Star_Wars_10distance[times])
	print(Star_Wars_10index[times])
	for j in range(0,N2):
		movie_distance[times,j] = np.sqrt(np.sum((v[times,j,:] - GoodFellas)**2))
	GoodFellas_10index[times] = np.argsort(movie_distance[times])[1:11]
	GoodFellas_10distance[times] = np.sort(movie_distance[times])[1:11]
	for j in range(0,N2):
		movie_distance[times,j] = np.sqrt(np.sum((v[times,j,:] - My_Fair_Lady)**2))
	My_Fair_Lady_10index[times] = np.argsort(movie_distance[times])[1:11]
	My_Fair_Lady_10distance[times] = np.sort(movie_distance[times])[1:11]




rows = ['objective function','rmse']
fig,ax = plt.subplots()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
the_table = ax.table(cellText = np.concatenate((L[:,99].reshape((1,10)),rmse[:,99].reshape((1,10)))),
	rowLabels = rows,
	loc = 'center'
	)
plt.show()

colorset = ['b','r','c','m','g','y','k','#fe1abf','#4991f1','#9a4aca']
timeset = ['first','second','third','forth','fifth','sixth','seventh','eighth','nineth','tenth']
rounds = np.linspace(2,101,99)
for k in range(0,10):
	plt.plot(rounds,L[k,1:],colorset[k],label=timeset[k]+' time')
plt.xlabel('iteration')
plt.ylabel('objective function')
plt.legend()
plt.show()







