import numpy as np
from chebyquad_problem import chebyquad, gradchebyquad  # ← 修正函数名
from base import OptimizationProblem

def make_chebyquad_problem(n: int, use_grad: bool = True):
    dim = n + 1

    def f(x: np.ndarray) -> float:
        # 老师实现依赖 len(x) = dim
        return chebyquad(x)

    def g(x: np.ndarray) -> np.ndarray:
        return gradchebyquad(x)

    # 经典稳健初值：x_j = (j+1)/(dim+1) ∈ (0,1)
    x0 = np.linspace(1, dim, dim, dtype=float) / (dim + 1.0)

    if use_grad:
        problem = OptimizationProblem(f, g)       # ← 只传 function, gradient
    else:
        problem = OptimizationProblem(f, None)    # 自动用数值梯度

    return problem, x0