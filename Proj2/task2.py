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

true_start = time.time()
#Load de dados
mat_file = 'data3.mat'

data = sio.loadmat(mat_file)
X = data['X']
Y = data['Y']

#Variáveis gerais
alpha_hat = 1
gama = 1e-4
beta = 0.5
n = X.shape[0]
K = X.shape[1]
epsilon = 1e-6

#Funções de cálculo de f(s,r)
def f(s,r,x,y,size):
    final = 0
    for a in range(size):
        final += np.log(1 + np.exp(s@x[:,a] - r)) - y[:,a]*(s@x[:,a] - r)
    return final/size

iterations = 0
s = -np.ones(n)
r = 0.0
grad = []
#Assinatura de tempo quando começou
start = time.time()
while True:
    df_ds = 0
    df_dr = 0
    for k in range(K):
        df_ds += (X[:,k]*np.exp(s @ X[:,k] - r))/(1 + 
                            np.exp(s @ X[:,k] - r)) - Y[:,k]*X[:,k]
        
        df_dr += (-np.exp(s @ X[:,k] - r))/(1 + 
                            np.exp(s @ X[:,k] - r)) + Y[:,k]
    df_ds /= K
    df_dr /= K        
    g = float(np.sqrt(df_dr**2 + np.linalg.norm(df_ds)**2))
    grad.append(g)
    #print('Grad =',g)
    if g < epsilon:
        break
    dk = -df_ds
    dk = np.append(dk, -df_dr, axis=0) 
    #Backtracking routine
    alpha_k = float(alpha_hat)
    while True:
        sk = s + alpha_k*(-df_ds)
        rk = r + alpha_k*(-df_dr)
        f_next = f(sk,rk,X,Y,K)
        f_current = f(s,r,X,Y,K) + gama*alpha_k*(-dk)@(dk)
        if  f_next < f_current:
            s = sk
            r = rk
            break
        else:
            alpha_k *= beta
    iterations += 1
#Assinatura de tempo em que terminou
end = time.time()
print('Tempo (Gradient Descent) =',datetime.timedelta(seconds = end - start))
#Equação da reta obtida
"""
x2 = lambda s_in, r_in, x1: (-s[0]*x1 + r_in)/s[1]

plt.figure()
plt.plot(range(iterations+1), grad)
plt.yscale('log')
plt.show()
plt.figure()
plt.scatter(X[0],X[1],c=Y)
reta = np.linspace(-6,6,1000)
plt.plot(reta,x2(s,r,reta))
plt.show()
"""
true_end = time.time()
print('Tempo total =',datetime.timedelta(seconds = true_end - true_start))