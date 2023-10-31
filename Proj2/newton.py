# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:38:32 2020

@author: Alexandre, Francisco, Miguel, Tiago
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time
import datetime

#Load de dados
mat_file = 'data4.mat'

data = sio.loadmat(mat_file)
X = data['X']
Y = data['Y']

#Variáveis gerais
alpha_hat = 1.0
gama = 1e-4
beta = 0.5
n = X.shape[0]
K = X.shape[1]
epsilon = 1e-6

#Funções de cálculo de f(s,r)

def f(a,x,y,size):
    final = sum(np.transpose(np.logaddexp(np.zeros(K),a@x) - y*(a@x)))/size
    return final

iterations = 0
sr = -np.ones(n)
sr = np.append(sr,0.0)
sr = np.reshape(sr,(n+1,))
grad = []
alfa_history = []

#Assinatura de tempo quando começou
start = time.time()
A = np.vstack((X,-np.ones(K)))
At = np.transpose(A)
f_forward = f(At,sr,Y,K)
while True:
    gk = sum(np.exp(At@sr)/(1 + np.exp(At@sr)) - Y)
    gk = A@gk/K
    df_dx_2 = (np.exp(At@sr))/((1 + np.exp(At@sr))**2)
    df_dx_2 = df_dx_2.flatten()
    
    D = np.diag(df_dx_2)
    H = A@D@At/K
    
    g = np.linalg.norm(gk)
    grad.append(g)
    #print('Grad =',g)
    if g < epsilon:
        break
    
    dk = -np.linalg.inv(H)@gk
    #Backtracking routine
    alpha_k = alpha_hat
    while True:
        srk = sr + alpha_k*dk
        f_next = f(At,srk,Y,K)
        f_current = f_forward + gama*gk@(alpha_k*dk)
        if  f_next < f_current:
            alfa_history.append(alpha_k)
            f_forward = f_next
            sr = srk
            break
        else:
            alpha_k *= beta
    iterations += 1
    
#Assinatura de tempo em que terminou
end = time.time()
print('Tempo (Newton) =',datetime.timedelta(seconds = end - start))

plt.figure()
plt.plot(range(1,len(grad)+1), grad)
plt.yscale('log')
plt.grid(linestyle='-',linewidth=.5)
plt.xlabel('Iterações')
plt.ylabel('||'r'$\nabla$ f(sk,rk)||')
plt.title('Módulo do gradiente por iteração do \
          método de Newton\n('+mat_file+')')
plt.show()
