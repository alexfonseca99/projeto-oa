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

#import random 
start = time.time()
k = 2
y_file = 'yinit{}.csv'.format(k)
data = np.genfromtxt('data_opt.csv', delimiter = ',')
y = np.genfromtxt(y_file, delimiter = ',')
y = np.reshape(y, (len(y), 1)) #converter para ter dimensão N,1 em vez de N,
lambda_0 = 1

#Task 1
D = spatial.distance.cdist(data, data, 'euclidean')
size = len(D)

maximum_dist = np.amax(D)
maximum_dist_index = []
maximum_dist_index.append(np.argmax(D) // size + 1) 
maximum_dist_index.append(np.argmax(D) % size + 1)
#+1 para ficar com a mesma convençao que o enunciado onde os indices começam em 1
print('Distancia máxima =',maximum_dist,'em D(',maximum_dist_index[0],',',
      maximum_dist_index[1],')\n')


grad_f_history = []
cost_history = []
epsilon = k*1e-2
lambda_k = lambda_0
iterations = -1
grad_f = 1

while grad_f > epsilon:

    list_grad_fnm = []
    list_fnm_values = []
    list_grad_fnm_square = []
    cost = 0
    for m in range(0, size*k, k):
        for n in range(m + k, size*k, k):
            norm = []
            grad_fnm = np.zeros(size*k)
            grad_fnm_square = np.zeros(size*k)
            for i in range(k):
                norm.append(y[m+i]-y[n+i])
            norm = np.linalg.norm(norm)
            
            for i in range(k):
                grad_fnm[m+i] = (y[m+i]-y[n+i])/norm
                grad_fnm[n+i] = -grad_fnm[m+i]
                grad_fnm_square[m+i] = 2*(y[m+i]-y[n+i]) - 2*D[int(m/k), int(n/k)]*grad_fnm[m+i]
                grad_fnm_square[n+i] = 2*(y[n+i]-y[m+i]) - 2*D[int(m/k), int(n/k)]*grad_fnm[n+i]
                
            cost += (norm - D[int(m/k), int(n/k)])**2
            list_fnm_values.append(norm - D[int(m/k), int(n/k)])
            list_grad_fnm.append(grad_fnm)
            list_grad_fnm_square.append(grad_fnm_square)
    
    list_fnm_values = np.array(list_fnm_values)
    list_fnm_values = np.reshape(list_fnm_values,(len(list_fnm_values),1))
    A = np.vstack((list_grad_fnm, np.sqrt(lambda_k)*np.identity(size*k)))
    b = np.vstack((list_grad_fnm @ y - list_fnm_values, np.sqrt(lambda_k)*y))
    ls = np.linalg.lstsq(A, b)
    y_ls = ls[0]
    grad_f = np.linalg.norm(sum(list_grad_fnm_square))
    grad_f_history.append(grad_f)
    cost_ls = 0
    
    for m in range(0, size*k, k):
        for n in range(m + k, size*k, k):
            norm = []
            for i in range(k):
                norm.append(y_ls[m+i] - y_ls[n+i])
            norm = np.linalg.norm(norm)
            
            cost_ls += (norm - D[int(m/k), int(n/k)])**2

    if cost_ls < cost:
        y = y_ls
        lambda_k *= 0.7
        cost_history.append(cost_ls)
    else:
        lambda_k *= 2
    
    iterations += 1
    
plt.figure()
plt.grid()
plt.plot(range(len(grad_f_history)),grad_f_history[0:len(grad_f_history)], marker='o', markersize=3)
plt.yscale('log')
plt.xlabel('Iteração')
plt.ylabel('||'r'$\nabla$f(y)||')
plt.xticks(range(0,iterations+1,20))
plt.title('Módulo do gradiente da função de custo ao\nlongo das iterações do algoritmo')
plt.show()
        





















