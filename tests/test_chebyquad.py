

import inspect
import numpy as np
from time import perf_counter

# ---- problems & baselines ----
from problems.chebyquad_adapter import make_chebyquad_problem
from chebyquad_problem import chebyquad, gradchebyquad
from scipy.optimize import fmin_bfgs

# ---- our optimizers ----
from Optimizers import NewtonOptimizer
from QNOptimizers import BFGS, DFP


# 三个类名
try:
    from QNOptimizers import BroydenGood
except Exception:
    BroydenGood = None
try:
    from QNOptimizers import BroydenBad
except Exception:
    BroydenBad = None
try:
    from QNOptimizers import SymmetricBroyden
except Exception:
    SymmetricBroyden = None


# --------- 公共工具 ---------
def call_with_supported_args(cls, *args, **kwargs):
    """根据构造函数签名，丢弃不被支持的关键字参数，防止 TypeError。"""
    sig = inspect.signature(cls)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(*args, **supported)

def run_ours(method_cls, name, n, tol=1e-8, max_iter=2000, **extra_kwargs):
    """运行我们的优化器，返回统一记录。"""
    problem, x0 = make_chebyquad_problem(n, use_grad=True)
    optimizer = call_with_supported_args(
        method_cls, problem, tol=tol, max_iter=max_iter, **extra_kwargs
    )
    t0 = perf_counter()
    x_star = optimizer.solve(x0)
    t1 = perf_counter()

    # 统一提取迭代信息
    iters = None
    if hasattr(optimizer, "iter_info") and isinstance(optimizer.iter_info, dict):
        iters = optimizer.iter_info.get("iter", None)

    f_star = problem.f(x_star)
    g_norm = np.linalg.norm(gradchebyquad(x_star))
    ok = np.isfinite(f_star) and np.isfinite(g_norm)
    return dict(method=name, n=n, iters=iters, fval=f_star, grad_norm=g_norm,
                time=t1 - t0, ok=ok)

def run_scipy_bfgs(n, gtol=1e-8, maxiter=2000):
    """scipy 基线：fmin_bfgs（无约束）。"""
    _, x0 = make_chebyquad_problem(n, use_grad=True)
    f, g = chebyquad, gradchebyquad
    t0 = perf_counter()
    x_star = fmin_bfgs(f, x0, fprime=g, gtol=gtol, maxiter=maxiter, disp=False)
    t1 = perf_counter()
    f_star = f(x_star)
    g_norm = np.linalg.norm(g(x_star))
    return dict(method="scipy-fmin_bfgs", n=n, iters=None, fval=f_star,
                grad_norm=g_norm, time=t1 - t0, ok=True)

def pretty_print(rows):
    header = ["n", "method", "iters", "f(x*)", "||grad||", "time(s)", "ok"]
    print("{:>3}  {:>20}  {:>6}  {:>12}  {:>10}  {:>8}  {:>3}".format(*header))
    for r in rows:
        it = "-" if r["iters"] is None else r["iters"]
        print("{:>3}  {:>20}  {:>6}  {:>12.4e}  {:>10.2e}  {:>8.3f}  {:>3}".format(
            r["n"], r["method"], it, r["fval"], r["grad_norm"], r["time"], "✓" if r["ok"] else "×"
        ))