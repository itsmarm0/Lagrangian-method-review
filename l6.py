import numpy as np
import sympy as sp
from scipy.optimize import fsolve

def func(X) :
    x = X[0]
    y = X[1]
    z = X[2]
    l1 = X[3] # This is the first multiplier
    l2 = X[4] # This is the second multiplier

    return x + y + z + l1 * (x**2 + y - 3) + l2 * (x + 3*y +2*z - 7)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3 # This is the step size used in the finite difference
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda


X1 = fsolve(dfunc, [0.5, 0.5, 0.5, 0.5, 0.5])
print (X1, func(X1))



x1, x2, x3, lambda1, lambda2 = sp.symbols('x1 x2 x3 lambda1 lambda2')


# Lagrange func
L = x1 + x2 + x3 + lambda1 * (x1**2 + x2 - 3) + lambda2 * (x1 + 3*x2 + 2*x3 - 7)

# Gradient
grad_L = [sp.diff(L, var) for var in (x1, x2, x3, lambda1, lambda2)]

# Hessian matrix
jacobian_L = sp.Matrix(grad_L).jacobian([x1, x2, x3, lambda1, lambda2])


grad_L_func = sp.lambdify((x1, x2, x3, lambda1, lambda2), grad_L, 'numpy')
jacobian_L_func = sp.lambdify((x1, x2, x3, lambda1, lambda2), jacobian_L, 'numpy')


def newton_raphson(x0, tolerance=1e-3, max_iterations=3):
    x_n = np.array(x0, dtype=float)
    for i in range(max_iterations):
        grad = np.array(grad_L_func(*x_n), dtype=float).flatten()
        H = np.array(jacobian_L_func(*x_n), dtype=float)
        if np.linalg.norm(grad, ord=2) < tolerance:
            print(f"Converged to: {x_n}")
            L_opt = lagrangian(*x_n)
            print(f"Optimal Lagrangian value: {L_opt}")
            return x_n
        delta_x = np.linalg.solve(H, -grad)
        x_n = x_n + delta_x
        print(f"Iteration {i + 1}: {x_n}")
    L_opt = lagrangian(*x_n)
    print("Did not fully converge after the maximum number of iterations")
    print(f"Optimal Lagrangian value: {L_opt}")
    return x_n


lagrangian = sp.lambdify((x1, x2, x3, lambda1, lambda2), L, 'numpy')

# Initial guess
x0 = [0.5, 0.5, 0.5, 0.5, 0.5]


result = newton_raphson(x0)
print("Result:", result)