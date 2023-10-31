# -*- coding: utf-8 -*-


import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

t_estrela = 8
K = 5
t = np.array([0, 1, 1.5, 3, 4.5])

ck = np.array([[0.6332, -0.0054, 2.3322, 4.4526, 6.1752],
               [-3.2012, -1.7104, -0.7620, 3.1001, 4.2391]])
               
Rk = np.array([2.2727, 0.7281, 1.3851, 1.81914, 1.0895])

po = cvx.Variable((2, 1))
v = cvx.Variable((2, 1))
direita = cvx.Variable(1)
esquerda = cvx.Variable(1)
cima = cvx.Variable(1)
baixo =  cvx.Variable(1)

constraints = []
direita_vec = []
esquerda_vec = []
cima_vec = []
baixo_vec = []
for i in range(K):
    constraints.append(cvx.pnorm(po + v * t[i] - np.reshape(ck[:, i], (2, 1)), 2) <= Rk[i])
    direita_vec.append(po[0] + v[0]*t_estrela + (ck[i] + Rk[i]))
    esquerda_vec.append(po[0] + v[0]*t_estrela - (ck[i] - Rk[i]))
    cima_vec.append(po[1] + v[1]*t_estrela + (ck[i] + Rk[i]))  
    baixo_vec.append(po[1] + v[1]*t_estrela - (ck[i] - Rk[i]))  

direita_vec = cvx.max(direita_vec)
esquerda_vec = cvx.max(esquerda_vec)    
cima_vec = cvx.max(cima_vec)    
baixo_vec = cvx.max(baixo_vec)        

if direita_vec < 2*(po[0] + v[0]*t_estrela):
    direita_vec = po[0] + v[0]*t_estrela
else:
    direita_vec = cvx.max(direita_vec) - po[0] + v[0]*t_estrela
    
if esquerda_vec < 2*(po[0] + v[0]*t_estrela):
    esquerda_vec = po[0] + v[0]*t_estrela
else:
    esquerda_vec = cvx.max(esquerda_vec) - (po[0] + v[0]*t_estrela)

if cima_vec < 2*(po[1] + v[1]*t_estrela):
    cima_vec = po[1] + v[1]*t_estrela
else:
    cima_vec = cvx.max(cima_vec) - po[1] + v[1]*t_estrela
    
if baixo_vec < 2*(po[1] + v[1]*t_estrela):
    baixo_vec = po[1] + v[1]*t_estrela
else:
    baixo_vec = cvx.max(baixo_vec) - po[1] + v[1]*t_estrela

cost = (direita - esquerda) * (cima - baixo)
problem = cvx.Problem(cost, )