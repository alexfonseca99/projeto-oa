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
#Mudar valor de k dependendo do ficheiro de dados a usar
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
maximum_dist_index.append(np.argmax(D) // size) 
maximum_dist_index.append(np.argmax(D) % size)
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
#Scatter dos dados para k=2
if k == 2:
    plt.figure()
    plt.grid()
    plt.scatter(y[0:len(y):k],y[k-1:len(y):k], s=8)
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.title('Dados com dimensão reduzida para k=2')
    plt.show()

#Função de custo ao longo do algoritmo

plt.figure()
plt.grid()
plt.plot(range(len(cost_history)),cost_history, marker='o', markersize=4)
plt.yscale('log')
plt.xlabel('Iteração')
plt.ylabel('Custo')
plt.title('Função de custo ao longo das iterações do algoritmo')
plt.show()

#Módulo do gradiente ao longo do algoritmo

plt.figure()
plt.grid()
plt.plot(range(len(grad_f_history)),grad_f_history[0:len(grad_f_history)],
         marker='o', markersize=4)
plt.yscale('log')
plt.xlabel('Iteração')
plt.ylabel('||'r'$\nabla$f(y)||')
plt.title('Módulo do gradiente da função de custo ao\nlongo das\
 iterações do algoritmo')
plt.show()

#Scatter dos dados para k=3
if k == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.grid()
    ax.scatter(y[0:len(y):k],y[1:len(y):k],y[k-1:len(y):k], s=8)
    ax.view_init(elev=30, azim=270)
    plt.xlabel('y1')
    plt.ylabel('y2')
    ax.set_zlabel('y3')
    plt.title('Dados com dimensão reduzida para k=3')
    plt.show()

