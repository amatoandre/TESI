# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:16:44 2023

@author: Andrea
"""

import math
import numpy as np 
import numpy.matlib as ml

# Metodo di Eulero X Metodo Monte Carlo

# N: numero di iterazioni (temporali)
# M: numero di simulazioni in ogni istante
# T: istante finale
# mu: funzione di Drift
# sigma: funzione di Diffuzione
# equazione dXt = ( alfa cos(Xt) - beta sen(Xt) ) dt + sigma dWt
# voglio MC di E[sen(Xt)]
# -- NB --
# usa OPERAZIONI VETTORIALI
# usa f1 per step di Eulero e f2 per il ciclo for
# -- --


def eulero_monte_carlo(c1, c2, sigma, T, N, M, X0):
   h = T / N
   X = ml.repmat(X0, 1, M)
   
   for i in range(N):
       W = np.random.normal(0, 1, (1, M)) 
       drift = c1 * np.cos(X) - c2 * np.sin(X) 
       diffusione = sigma 
       X = X + drift * h + diffusione * math.sqrt(h) * W 
       
   return X

# def eulero(c1, c2, sigma, h, M, X):
#        W = np.random.normal(0, 1, (1, M)) 
#        drift = c1 * np.cos(X) - c2 * np.sin(X) 
#        diffusione = sigma 
#        X = X + drift * h + diffusione * math.sqrt(h) * W 
#       
#    return X


def monte_carlo(M, Y):
    X = sum(Y) / M
    return X

if __name__ == "__main__":
    
    # Parametri in Input
    c1 = 1
    c2 = 0
    sigma = 0.5
    T = 10
    N = 2
    M = 10
    X0 = 100
    
    # Richiami funzioni
    XT = eulero_monte_carlo(c1, c2, sigma, T, N, M, X0)
    print(XT)
    
    YT = c1 * np.sin(XT) + c2 * np.cos(XT)
    YT = YT.transpose()
    F = monte_carlo(M, YT)
    print(F)
    


