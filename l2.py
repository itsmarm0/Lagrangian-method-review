import numpy as np
from scipy.optimize import fsolve

def func(X) :
    x = X[0]
    y = X[1]
    l = X[2] # This is the multiplier

    return 2*x + x*y + 3*y + l * (x**2 + y - 3)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3 # This is the step size used in the finite difference
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda

# This is the max
X1 = fsolve(dfunc, [1, 1, 0])
print (X1, func(X1))

# This is the min
X2 = fsolve(dfunc, [-1, -1, 0])
print (X2, func(X2))