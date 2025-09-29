import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base import OptimizationProblem
from Optimizers import NewtonOptimizer



def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosenbrock_grad(x):
    x = np.asarray(x, dtype=float)
    dfdx = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    dfdy = 200.0 * (x[1] - x[0] ** 2)
    return np.array([dfdx, dfdy], dtype=float)


def test_newton_exact_line_search_rosenbrock():
    prob = OptimizationProblem(rosenbrock, rosenbrock_grad)
    opt = NewtonOptimizer(prob, line_search_type='exact', tol=1e-6, max_iter=500)

    x0 = np.array([-1.2, 1.0], dtype=float)
    x_star = opt.solve(x0)

    # target point should be close to (1, 1)
    assert np.allclose(x_star, np.array([1.0, 1.0]), atol=1e-4)
    # norm should be small enough
    assert np.linalg.norm(prob.g(x_star)) < 1e-5



def test_newton_inexact_line_search_rosenbrock():
    prob = OptimizationProblem(rosenbrock, rosenbrock_grad)
    opt = NewtonOptimizer(prob, line_search_type='inexact', tol=1e-6, max_iter=500)

    x0 = np.array([-1.2, 1.0], dtype=float)
    x_star = opt.solve(x0)

    # 目标点接近 (1, 1)
    assert np.allclose(x_star, np.array([1.0, 1.0]), atol=1e-4)
    # 梯度范数应足够小
    assert np.linalg.norm(prob.g(x_star)) < 1e-5

