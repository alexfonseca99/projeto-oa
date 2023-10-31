# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

#Dados gerais
t_estrela = 8
K = 5
t = np.array([0, 1, 1.5, 3, 4.5])

#Exemplo dado para confirmar
"""
ck = np.array([[-1.7210, 1.0550, 2.9619, 3.8476, 7.1086],
               [-4.3454, -3.0293, -1.5857, 1.2253, 4.9975]])
               
Rk = np.array([0.9993, 1.4618, 2.2617, 1.0614, 1.6983])
"""
#Dados do exercício para resolver

ck = np.array([[0.6332, -0.0054, 2.3322, 4.4526, 6.1752],
               [-3.2012, -1.7104, -0.7620, 3.1001, 4.2391]])
               
Rk = np.array([2.2727, 0.7281, 1.3851, 1.81914, 1.0895])

#Definição do problema
po = cvx.Variable((2, 1))
v = cvx.Variable((2, 1))
a1 = cvx.Variable()
a2 = cvx.Variable()
b1 = cvx.Variable()
b2 = cvx.Variable()


constraints = []

for i in range(K):
    constraints.append(cvx.pnorm(po + v * t[i] - np.reshape(ck[:, i], (2, 1)), 2) <= Rk[i])

cost1 = po[0] + v[0]*t_estrela #x_final
cost2 = po[1] + v[1]*t_estrela #y_final

problem1 = cvx.Problem(cvx.Minimize(cost1), constraints)
problem2 = cvx.Problem(cvx.Maximize(cost1), constraints)
problem3 = cvx.Problem(cvx.Minimize(cost2), constraints)
problem4 = cvx.Problem(cvx.Maximize(cost2), constraints)

sols = []
sols.append(problem1.solve(verbose=True)) #a1 (esquerda)
sols.append(problem2.solve(verbose=True)) #a2 (direita)
sols.append(problem3.solve(verbose=True)) #b1 (baixo)
sols.append(problem4.solve(verbose=True)) #b2 (cima)


#Plots
#Definições dos sistema de eixos
fig, ax = plt.subplots()


ax.set_xlim((-4, 20)) #Limites do eixo x
ax.set_ylim((-6, 18)) #Limites do eixo y
ax.set_aspect('equal') #Meter a janela do gráfico quadrada para não deformar os circulos
ax.set_axisbelow(True) #Meter as linhas de guia atrás dos pontos do gráfico
plt.xticks(range(-4,20,2)) #Números representados no eixo x

#Pontos/circulos
for i in range(K): 
    ax.add_artist(plt.Circle((ck[0, i], ck[1, i]), Rk[i], fill=False, color='b'))

plt.scatter(sols[0], sols[2])
plt.scatter(sols[0], sols[3])
plt.scatter(sols[1], sols[2])
plt.scatter(sols[1], sols[3])

plt.grid()
plt.show()

print("Area = ", (sols[1]-sols[0])*(sols[3]-sols[2]))

    