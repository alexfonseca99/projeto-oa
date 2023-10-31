# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:15:49 2020

@author: Alexandre, Francisco, Miguel, Tiago
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import matplotlib.ticker as ticker
import time
import datetime
import random 
import pickle
start = time.time()
k = 2
data = np.genfromtxt('dataProj.csv', delimiter = ',')
D = spatial.distance.cdist(data, data, 'euclidean')
size = len(D)
y = np.random.uniform(low=-5000, high=-2500, size=(k*size,1))
#y = np.reshape(y, (len(y), 1)) #converter para ter dimensão N,1 em vez de N,
lambda_0 = 1

#Task 1


maximum_dist = np.amax(D)
maximum_dist_index = []
maximum_dist_index.append(np.argmax(D) // size + 1) 
maximum_dist_index.append(np.argmax(D) % size + 1)
#+1 para ficar com a mesma convençao que o enunciado onde os indices começam em 1
print('Distancia máxima =',maximum_dist,'em D(',maximum_dist_index[0],',',
      maximum_dist_index[1],')\n')

#Task 3
grad_f_history = []
cost_history = []
b = []
epsilon = k*1e-2
lambda_k = lambda_0
iterations = 0

while True:
    cost = 0
    b = []
    A = []
    list_grad_fnm = []
    list_grad_fnm_square = []
    
    for m in range(0, k*size-1, k):
        for n in range(m + k, k*size-1, k): 
            grad_fnm = np.zeros(k*size)
            grad_fnm_square = np.zeros(k*size)
            norm = []
            for i in range(k):
                #Componentes da norma
                norm.append(y[m+i]-y[n+i])
            norm = np.linalg.norm(norm)
            #Valor da função de custo
            cost += (norm - D[int(m/k), int(n/k)])**2
            for i in range(k):
                grad_fnm[m+i] = (y[m+i]-y[n+i])/norm
                grad_fnm[n+i] = -grad_fnm[m+i]
                grad_fnm_square[m+i] = 2*(y[m+i] - y[n+i]) - \
                2*D[int(m/k),int(n/k)]*grad_fnm[m+i]

                grad_fnm_square[n+i] = -2*(y[m+i] - y[n+i]) - \
                2*D[int(m/k),int(n/k)]*grad_fnm[n+i]
                
            list_grad_fnm.append(grad_fnm)
            list_grad_fnm_square.append(grad_fnm_square)
            b = np.append(b, grad_fnm@y - (norm - D[int(m/k),int(n/k)]))
    
    b = np.reshape(np.array(b), (len(b), 1))
    b = np.vstack((b, np.sqrt(lambda_k)*y))
    A = np.vstack((list_grad_fnm, np.sqrt(lambda_k)*np.identity(size*k)))
    ls = np.linalg.lstsq(A, b, rcond=None)
    cost_ls = 0
    for m in range(0, k*size-1, k):
        for n in range(m + k, k*size-1, k):
            norm = []
            for i in range(k):
                norm.append(ls[0][m+i]-ls[0][n+i])
            norm = np.linalg.norm(norm)
            cost_ls += (norm - D[int(m/k), int(n/k)])**2

    grad_f = np.linalg.norm(sum(list_grad_fnm_square))
    grad_f_history.append(grad_f)
    
    if grad_f < epsilon:
        break
    
    if cost_ls < cost:
        y = ls[0]
        lambda_k *= 0.7
        cost_history.append(cost_ls)
    else:
        lambda_k *= 2
    iterations += 1        

end = time.time()
print('Tempo (LM) =', datetime.timedelta(seconds = end - start))

plt.scatter(y[0::2],y[1::2])
#Função de custo ao longo do algoritmo para k=2

plt.figure()
plt.grid()
plt.plot(range(iterations),cost_history, marker='o')
plt.yscale('log')
plt.xlabel('Iteração')
plt.ylabel('Custo')
plt.xticks(range(0,iterations,4))
plt.title('Função de custo ao longo das iterações do algoritmo')
plt.show()

#Módulo do gradiente ao longo do algoritmo para k=2
plt.figure()
plt.grid()
plt.plot(range(iterations+1),grad_f_history[0:len(grad_f_history)], marker='o')
plt.yscale('log')
plt.xlabel('Iteração')
plt.ylabel('||'r'$\nabla$f(y)||')
plt.xticks(range(0,iterations+1,4))
plt.title('Módulo do gradiente da função de custo ao\nlongo das iterações do algoritmo')
plt.show()