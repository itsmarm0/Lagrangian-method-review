import numpy as np
import math
from scipy.optimize import fsolve

def func(X) :
    p1 = X[0]
    p2 = X[1]
    p3 = X[2]
    l = X[3] # This is the multiplier

    return math.log(1 + 0.9*p1) + math.log(0.9 + 0.8*p2) + math.log(1 + 0.7*p3) + l * (p1 + p2 + p3 - 1)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3 # This is the step size used in the finite difference
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda
    
# This the max    
X1 = fsolve(dfunc, [1, 1, 1, 0])
print (X1, func(X1))