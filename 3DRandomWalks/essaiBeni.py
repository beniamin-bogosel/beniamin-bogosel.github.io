#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 vincent <vincent@vincent-U36SG>
#
# Distributed under terms of the MIT license.

"""
Fonctions pour l'analyse des modèles de marches aléatoires.
"""

import sympy as sp
import numpy as np
import math
from scipy import optimize
from sympy.utilities.lambdify import lambdify


x,y,z = sp.symbols("x y z", positive=True)

def get_inventory(pas):
    """Calcul de la fraction génératrice du modèle."""
    resultat = 0
    for i,j,k in pas:
        resultat += x**i * y**j * z**k
    return resultat

def get_derives(inventory):
    """Calcul des dérivées."""
    return sp.diff(inventory, x), sp.diff(inventory, y), sp.diff(inventory, z)

def get_derives_v(inventory):
    """Calcul des dérivées."""
    return (sp.diff(inventory, x), sp.diff(inventory, y), sp.diff(inventory, z))


def get_zeros(derives):
    """Calcul du points critique."""
    return sp.solve(derives, (x,y,z))

def get_zeros_approx(derives):
    """Calcul approximatif du point critique."""
    return list(sp.nsolve(derives, (x,y,z), (0.9,0.9,0.9)))

def get_optim_approx(inventory):
    """optimization"""
    my_func = lambdify((x,y,z),inventory)
    grad    = get_derives_v(inventory) 
    j1 = lambdify((x,y,z),grad[0])
    j2 = lambdify((x,y,z),grad[1])
    j3 = lambdify((x,y,z),grad[2])
    def my_func_v(x):
       return my_func(*tuple(x))
    def my_jac_v(x):
       g  = [None]*3
       g[0] = j1(*tuple(x))
       g[1] = j2(*tuple(x))
       g[2] = j3(*tuple(x)) 
       return g
    print my_jac_v((1,1,1.2))
    bnds = ((0, None), (0, None),(0,None))
    results = optimize.minimize(my_func_v,[1,1,1],
                                bounds=bnds
                                )
    result = results.x
    return result

#list(optimize.minimize(my_func_v, (0.25*math.pi,0.25*math.pi,0.25*math.pi),
 #         method='L-BFGS-B'))

def get_matrice(inventory, zero):
    """Génération de la matrice de covariance."""
    x0, y0, z0 = zero
    valeurs = {x:x0, y:y0, z:z0}
    a = (sp.diff(inventory, x, y)/sp.sqrt(sp.diff(inventory, x, x)*
        sp.diff(inventory, y, y))).subs(valeurs)
    b = (sp.diff(inventory, x, z)/sp.sqrt(sp.diff(inventory, x, x)*
        sp.diff(inventory, z, z))).subs(valeurs)
    c = (sp.diff(inventory, y, z)/sp.sqrt(sp.diff(inventory, y, y)*
        sp.diff(inventory, z, z))).subs(valeurs)

    return sp.Matrix([[1, a, b], [a, 1, c], [b, c, 1]])

def get_matrice_approx(inventory, zero):
    """Génération de la matrice de covariance."""
    x0, y0, z0 = zero
    valeurs = {x:x0, y:y0, z:z0}
    a = (sp.diff(inventory, x, y)/sp.sqrt(sp.diff(inventory, x, x)*
        sp.diff(inventory, y, y))).subs(valeurs).evalf()
    b = (sp.diff(inventory, x, z)/sp.sqrt(sp.diff(inventory, x, x)*
        sp.diff(inventory, z, z))).subs(valeurs).evalf()
    c = (sp.diff(inventory, y, z)/sp.sqrt(sp.diff(inventory, y, y)*
        sp.diff(inventory, z, z))).subs(valeurs).evalf()

    return np.matrix([[1, a, b], [a, 1, c], [b, c, 1]], dtype=np.float)

def racine(D):
    """Calcul de la racine carrée pour une matrice diagonale"""
    return sp.Matrix([[sp.sqrt(D[0,0]),0,0],[0, sp.sqrt(D[1,1]),0],[0,0,sp.sqrt(D[2,2])]])
    
def racine_approx(D):
    """Calcul de la racine carrée pour une matrice diagonale"""
    return sp.Matrix([[sp.sqrt(D[0]),0,0],[0, sp.sqrt(D[1]),0],[0,0,sp.sqrt(D[2])]])

def ps(U,V):
    """produit scalaire de deux listes de tailles 3"""
    return sum([u*v for u,v in zip(U,V)])

def angle(u,v):
    """Calcul d'angle entre deux vecteurs."""
    return sp.acos(ps(u,v)/(sp.sqrt(ps(u,u))*sp.sqrt(ps(v,v))))

def main(pas, decimales=10):
    """Assemblage de l'algorithme"""
    inventory = get_inventory(pas)
    derives = get_derives(inventory)
    solutions = get_zeros(derives)
    print(solutions)
    if len(solutions) != 1:
        raise ValueError("Modèle dégénéré")
    matrice = get_matrice(inventory, solutions[0])
    angles = [sp.acos(-matrice[0,1]),sp.acos(-matrice[0,2]),sp.acos(-matrice[1,2])]
    return angles


def main_approx(pas):
    """Assemblage de l'algorithme"""
    inventory = get_inventory(pas)
    derives = get_derives(inventory)
    solutions = get_zeros_approx(derives)
    #print(solutions)
    matrice = get_matrice_approx(inventory, list(solutions))
    print(matrice)
    D, P = np.linalg.eig(matrice)
    R = P*np.diag([1/np.sqrt(x) for x in D])*np.linalg.inv(P)
    angles = [angle(R[:,0], R[:,1]), angle(R[:,0],R[:,2]), angle(R[:,1],R[:,2])]
    return angles

def main_approx2(pas):
    """Assemblage de l'algorithme"""
    inventory = get_inventory(pas)
    derives   = get_derives(inventory)
    #solutions = get_zeros_approx(derives)
    solutions      = get_optim_approx(inventory)
    #print solutions
    #print sols
    #print(solutions)
    matrice = get_matrice(inventory, list(solutions))
    #print(matrice)
    #print(matrice[1,2])
    #angles = [sp.acos(-matrice[0,1]),sp.acos(-matrice[0,2]),sp.acos(-matrice[1,2])]
    angles = [math.acos(-matrice[0,1]),math.acos(-matrice[0,2]),math.acos(-matrice[1,2])] 
    return angles




essai_normal = ((1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1))

#f = open('ListG1_nKrew3.csv', 'r')
#f = open('ListG4_nKrew3.csv', 'r')
#f = open('ListG6_nra1.csv', 'r')
f = open('ListG3_nra3.csv', 'r')
XX = f.readlines()

Pi = math.pi

print Pi

for i in range(0,len(XX)):
  print 'Iteration ', i, ' out of ', len(XX)
  exec('mat = ' + XX[i])
  ##print(mat)
  ang = main(mat)
  print 'Angles ', 1/Pi*np.asarray(ang)
  print ' '


