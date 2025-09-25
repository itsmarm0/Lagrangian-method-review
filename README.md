Lagrangian Methods in Nonlinear Programming

Overview:

This project provides a review and implementation of Lagrangian methods in nonlinear programming. It explores the mathematical foundations, presents algorithms, and compares the Lagrangian approach with Newton–Raphson and fsolve methods.
The project includes both the theoretical review (in Persian, PDF) and Python code implementations with numerical experiments.

Contents:

Report: A detailed review of Lagrangian methods (in Persian).
Code: Python scripts implementing Lagrangian multipliers with numerical solutions.

Comparisons between:

1) Lagrangian multipliers
2) Newton–Raphson method
3) scipy.optimize.fsolve

Methods:

Lagrangian Multipliers: Used for solving constrained optimization problems.
Newton–Raphson: Iterative method with fast convergence (requires accurate derivatives).
fsolve: A flexible solver from SciPy that automates derivative handling.

Results:

Newton–Raphson: Faster convergence when initial points are well chosen but highly sensitive to them.

fsolve: More stable and user-friendly but may require more iterations.

Lagrangian Method: Direct approach for constrained problems; stability depends on the problem structure.
