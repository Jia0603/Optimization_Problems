import numpy as np
from chebyquad_problem import chebyquad, gradchebyquad
from base import OptimizationProblem

def make_chebyquad_problem(n: int, use_grad: bool = True):
    dim = n + 1

    def f(x: np.ndarray) -> float:
        return chebyquad(x)

    def g(x: np.ndarray) -> np.ndarray:
        return gradchebyquad(x)

    # Classic Robust Initial Value：x_j = (j+1)/(dim+1) ∈ (0,1)
    x0 = np.linspace(1, dim, dim, dtype=float) / (dim + 1.0)

    if use_grad:
        problem = OptimizationProblem(f, g)       # ←  function, gradient
    else:
        problem = OptimizationProblem(f, None)    # Automatically use numerical gradient

    return problem, x0