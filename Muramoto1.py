# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:13:37 2023

@author: Andrea
"""
import math
import numpy as np 
import numpy.matlib as ml

# Metodo di Eulero X Metodo Monte Carlo

# N: numero di iterazioni (steps temporali)
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


def eulero(c1, c2, sigma, h, M, X):
    W = np.random.normal(0, 1, (1, M)) 
    drift = c1 * np.cos(X) - c2 * np.sin(X) 
    diffusione = sigma 
    X = X + drift * h + diffusione * math.sqrt(h) * W 
    
    return X


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
    
    # Richiamo funzione
    h = T / N
    X = ml.repmat(X0, 1, M)
    
    for i in range(N):
        XT = eulero(c1, c2, sigma, h, M, X)
        
        YT = c1 * np.sin(XT) + c2 * np.cos(XT)
        YT = YT.transpose()
        F = monte_carlo(M, YT)
    print(XT)
    print(F)