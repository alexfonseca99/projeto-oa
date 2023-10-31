# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

#Dados gerais
x_estrela = np.array([6,
                      10])
t_estrela = 8
K = 5
t = np.array([0, 1, 1.5, 3, 4.5])

#Exemplo dado para confirmar

ck = np.array([[-1.7210, 1.0550, 2.9619, 3.8476, 7.1086],
               [-4.3454, -3.0293, -1.5857, 1.2253, 4.9975]])
               
Rk = np.array([0.9993, 1.4618, 2.2617, 1.0614, 1.6983])

#Dados do exercício para resolver
"""
ck = np.array([[0.6332, -0.0054, 2.3322, 4.4526, 6.1752],
               [-3.2012, -1.7104, -0.7620, 3.1001, 4.2391]])
               
Rk = np.array([2.2727, 0.7281, 1.3851, 1.81914, 1.0895])
"""
#Definição do problema
po = cvx.Variable((2, 1))
v = cvx.Variable((2, 1))

constraints = []
for i in range(K):
    constraints.append(cvx.pnorm(po + v * t[i] - np.reshape(ck[:, i], (2, 1)), 2) <= Rk[i])

cost = cvx.pnorm(np.reshape(x_estrela, (2,1)) - po - v * t_estrela, 2)

problem = cvx.Problem(cvx.Minimize(cost), constraints)
problem.solve(verbose=True)
 

pfinal = [po[0].value + t_estrela*v[0].value, po[1].value + t_estrela*v[1].value]


#Plots
#Definições dos sistema de eixos
fig, ax = plt.subplots()
ax.set_xlim((-6, 12)) #Limites do eixo x
ax.set_ylim((-6, 12)) #Limites do eixo y
ax.set_aspect('equal') #Meter a janela do gráfico quadrada para não deformar os circulos
ax.set_axisbelow(True) #Meter as linhas de guia atrás dos pontos do gráfico
plt.xticks(range(-6,12,2)) #Números representados no eixo x

#Pontos/circulos
for i in range(K): #Adicionar os círculos ao gráfico
    ax.add_artist(plt.Circle((ck[0, i], ck[1, i]), Rk[i], fill=False, color='b'))
    plt.scatter(po[0].value + v[0].value * t[i], po[1].value + v[1].value * t[i],
                marker='s', facecolors='none', edgecolors='r')



plt.scatter(x_estrela[0], x_estrela[1], color='000000') #Ponto x*
plt.scatter(po[0].value + v[0].value * t_estrela, po[1].value + v[1].value * t_estrela,
             marker='s', facecolors='none', edgecolors='r') 

ax.annotate('x*', (x_estrela[0]-0.5, x_estrela[1]-1.25))
plt.grid()
plt.show()



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        