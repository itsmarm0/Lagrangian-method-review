Lagrangian Methods in Nonlinear Programming

Overview:

This project is a review on implementation of Lagrangian methods in nonlinear programming. It compares the Lagrangian approach with Newton–Raphson and fsolve methods.
It includes both the theoretical review (in Persian, PDF) and Python code implementations with numerical experiments.

Contents:
Report: A detailed review of Lagrangian methods (in Persian).
Code: Python scripts implementing Lagrangian multipliers with numerical solutions.

Comparisons between:
1) Lagrangian multipliers
2) Newton–Raphson method

Methods:
Lagrangian Multipliers: Used for solving constrained optimization problems.
Newton–Raphson: Iterative method with fast convergence (requires accurate derivatives).
fsolve: A flexible solver from SciPy that automates derivative handling.

Results:
Newton–Raphson: this algorithm has faster convergence when initial points are well chosen but also highly sensitive to them.
fsolve: It's More stable and user-friendly but may require more iterations.
Lagrangian Method: This algorithm has a direct approach for constrained problems; Its stability depends on the problem structure.
