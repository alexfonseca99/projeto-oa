# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

#Dados gerais
T = 80
t2 = np.arange(T)
reg_parameter = 0.1
U_max = 100 
K = 6
A = np.array([[1, 0, 0.1, 0],
              [0, 1, 0, 0.1],
              [0, 0, 0.9, 0],
              [0, 0, 0, 0.9]])

B = np.array([[0, 0],
              [0, 0],
              [0.1 ,0],
              [0, 0.1]])

E = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

#Dados dos estados
x_initial = np.array([0,
                      5,
                      0,
                      0])

x_final = np.array([15,
                    -15,
                    0,
                    0])

#Dados dos waypoints
#Fazer matriz com waypoints e array com tempos ou fazer separados?
w = np.array([[10, 20, 30, 30, 20, 10],
              [10, 10, 10, 0, 0, -10]])

tao = np.array([10, 25, 30, 40, 50, 60])


#Definição do problema
x = cvx.Variable((4, T + 1))
u = cvx.Variable((2, T))


cost1 = 0
cost2 = 0

constraints = [x[:, 0] == x_initial, 
               x[:, T] == x_final]

for i in range(T):
    constraints.append(x[:, i+1] == A @ x[:, i] + B @ u[:, i])
    
for i in range(1, T):
    constraints.append(cvx.pnorm(u[:,i], 2) <= U_max)
    cost2 += cvx.square(cvx.pnorm(u[:,i] - u[:, i-1], 2))

for i in range(K):
    cost1 += cvx.square(cvx.pnorm(E @ x[:, tao[i]] - w[:,i], 2))

    

cost = cost1 + reg_parameter * cost2


problem = cvx.Problem(cvx.Minimize(cost), constraints)

problem.solve(verbose=True)

#plot
plt.scatter(x[0].value, x[1].value, marker='o', facecolors='none', edgecolors='b', s=8)
plt.scatter(w[0,:], w[1,:], marker='s', facecolors='none', edgecolors='r')
plt.scatter(x[0, tao].value, x[1, tao].value, marker='o', facecolors='none', edgecolors='m', s=85)
plt.axis([0, 35, -15, 15])
plt.grid()
plt.show()


plt.plot(t2, u[0].value)
plt.plot(t2, u[1].value)
plt.axis([0, 80, -40, 40])
plt.grid()
plt.show()


deviation = 0
for i in range(K):
    deviation += np.linalg.norm(E @ x[:,tao[i]].value - w[:,i])
    
print('Mean deviation=',(1/K)*deviation)