import numpy as np
import matplotlib.pyplot as plt

with open('COMS4721_hw4-data/movies.txt') as f:
	lines = f.read().splitlines()


for i in range(1682):
	if "Star Wars" in lines[i]:
		print("Star Wars %d" %i)
	if "My Fair Lady" in lines[i]:
		print("My Fair Lady %d" %i)
	if "GoodFellas" in lines[i]:
		print("GoodFellas %d" %i)

u = np.zeros((10,943,10))
v = np.zeros((10,1682,10))
star_war = v[0,49,:]
distance = np.zeros(1682)
for i in range(0,1682):
	distance[i] = np.sqrt(np.sum((v[0,i,:] - star_war)**2))
xx = np.random.normal(0,0.1,1682)
xxx = np.sort(xx)
yyy = np.argsort(xx)
print(xxx[:10])
print(yyy[:10])
print(xx[yyy[:10]])


'''
rmse = np.zeros((10,100))
objective = np.ones((10,100))
rows = ['objective function','rmse']
fig,ax = plt.subplots()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
print(np.concatenate((objective[:,99].reshape((10,1)),rmse[:,99].reshape((10,1))),axis=1))
the_table = ax.table(cellText = np.concatenate((objective[:,99].reshape((1,10)),rmse[:,99].reshape((1,10)))),
	rowLabels = rows,
	loc = 'center'
	)
plt.show()

L = np.zeros((10,100))
colorset = ['b','r','c','m','g','y','k','#fe1abf','#4991f1','#9a4aca']
timeset = ['first','second','third','forth','fifth','sixth','seventh','eighth','nineth','tenth']
rounds = np.linspace(2,101,99)
for k in range(0,10):
	plt.plot(rounds,L[k,1:],colorset[k],label=timeset[k]+' time')
plt.xlabel('iteration')
plt.ylabel('objective function')
plt.legend()
plt.show()
'''