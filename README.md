# Optimization_Problems

Implementation of classical and quasi-Newton optimization algorithms for NUMN21/FMNN25: Advanced Numerical Algorithms in Python (Project 2).

The project covers Newton’s method (with exact and inexact line search), Quasi-Newton updates (DFP, BFGS, Broyden family), and tests on Rosenbrock and Chebyquad benchmark problems.

## Project Structure

    project02/
    ├── __init__.py
    ├── base.py            # Base classes: OptimizationProblem and Optimizer (Task 1 & 2)
    ├── Optimizers.py      # Newton's method implementations (Task 3, 4, 6, 8)
    ├── QNOptimizers.py    # Quasi-Newton methods: DFP, BFGS, Symmetric/Good/Bad Broyden (Task 9)
    ├── Implementation.ipynb  # Notebook with experiments and visualizations
    └── tests/
     ├── __init__.py
     ├── test_rosenbrock.py   # Verification on Rosenbrock (Task 5 & 7)
     ├── test_chebyquad.py    # Experiments with Chebyquad (Task 10 & 11)
     └── benchmark.py         # Evaluation tools for Hessian inverse quality (Task 12)
## !!! Important Naming Considerations !!!
In the five implementations of the QuasiNewtonOptimizer class, the state update and initialization function names inherit from the Optimizer base class: update_algorithm_state(self, x_old, x_new, g_old, dir, alpha); initialize_algorithm(self, x0); The dictionary for state storage is named: self.algorithm_state = {};

## Group contributions:
Zhe Zhang: Implemented the Newton’s method with exact line search method; Test the performance of this method on the Rosenbrock function and visualize the iterative trajectory.

Jiazhuang Chen: Implemented the chebyquad problem interface (task 10), ran experiments on chebyquad for n=4,8,11 using Newton and Quasi-Newton methods and compared with SciPy baseline (task 11), and studied the quality of BFGS inverse Hessian approximation with numerical Hessians and plotted error curves (task 12).

Ruizhen Shen:

Jiuen Feng: Implemented the Newton's method with inexact line search method and tested the performence on the Rosenbrock function.

Jia Gu: Built the two base classes in task 1 and 2; Implemented the Classic Netwon Method in task3.
