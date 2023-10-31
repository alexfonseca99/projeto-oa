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
    final = sum(np.transpose(np.logaddexp(np.zeros(K),a@x) - \
                             y*(a@x)))/size #multiplicaçao final errada
    return final

iterations = 0
sr = -np.ones(n)
sr = np.append(sr,0.0)
sr = np.reshape(sr,(n+1,))
grad = []

A = np.vstack((X,-np.ones(K)))
At = np.transpose(A)
f_forward = f(At,sr,Y,K)
#Assinatura de tempo quando começou
start = time.time()
while True:
    gk = sum(np.exp(At@sr)/(1 + np.exp(At@sr)) - Y)
    gk = A@gk/K     
    
    g = np.linalg.norm(gk)
    grad.append(g)
    #print('Grad =',g)
    if g < epsilon:

        break
    dk = -gk
    #Backtracking routine
    alpha_k = alpha_hat
    while True:
        srk = sr + alpha_k*dk
        f_next = f(At,srk,Y,K)
        f_current = f_forward + gama*alpha_k*(-dk)@gk
        if  f_next < f_current:
            f_forward = f_next
            sr = srk;
            break
        else:
            alpha_k *= beta
    iterations += 1
    
#Assinatura de tempo em que terminou
end = time.time()
print('Tempo (Gradient Descent) =',datetime.timedelta(seconds = end - start))

plt.figure()
plt.plot(range(iterations+1), grad)
plt.yscale('log')
plt.grid(linestyle='-',linewidth=.5)
plt.show()
"""
x2 = lambda s_in, r_in, x1: (-s[0]*x1 + r_in)/s[1]
plt.figure()
plt.scatter(X[0],X[1],c=Y)
reta = np.linspace(-6,6,1000)
plt.plot(reta,x2(s,r,reta))
plt.show()
"""