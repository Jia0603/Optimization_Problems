

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


# classes
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
    """Discard unsupported keyword arguments based on the constructor signature to prevent TypeError."""
    sig = inspect.signature(cls)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(*args, **supported)

def run_ours(method_cls, name, n, tol=1e-8, max_iter=2000, **extra_kwargs):
    '''Run our optimizer and return a unified record.'''
    problem, x0 = make_chebyquad_problem(n, use_grad=True)
    optimizer = call_with_supported_args(
        method_cls, problem, tol=tol, max_iter=max_iter, **extra_kwargs
    )
    t0 = perf_counter()
    x_star = optimizer.solve(x0)
    t1 = perf_counter()

    # Unified extraction of iteration information
    iters = None
    if hasattr(optimizer, "iter_info") and isinstance(optimizer.iter_info, dict):
        iters = optimizer.iter_info.get("iter", None)

    f_star = problem.f(x_star)
    g_norm = np.linalg.norm(gradchebyquad(x_star))
    ok = np.isfinite(f_star) and np.isfinite(g_norm)
    return dict(method=name, n=n, iters=iters, fval=f_star, grad_norm=g_norm,
                time=t1 - t0, ok=ok)

def run_scipy_bfgs(n, gtol=1e-8, maxiter=2000):
    """scipy base line：fmin_bfgs"""
    _, x0 = make_chebyquad_problem(n, use_grad=True)
    f, g = chebyquad, gradchebyquad
    t0 = perf_counter()
    x_star = fmin_bfgs(f, x0, fprime=g, gtol=gtol, maxiter=maxiter, disp=False)
    t1 = perf_counter()
    f_star = f(x_star)
    g_norm = np.linalg.norm(g(x_star))
    return dict(method="scipy-fmin_bfgs", n=n, iters=None, fval=f_star,
                grad_norm=g_norm, time=t1 - t0, ok=True)


def pretty_print(rows, tol_grad=1e-6, tol_f=1e-12, max_iters=2000, show_colors=True):
    """
    rows: list of dicts with keys:
      n, method, iters (int or None), fval, grad_norm, time, [optional] df
    """
    # --- optional color support ---
    if show_colors:
        try:
            from colorama import Fore, Style, init as colorama_init
            colorama_init(autoreset=True)
            GREEN, RED, RESET, CYAN = Fore.GREEN, Fore.RED, Style.RESET_ALL, Fore.CYAN
        except Exception:
            GREEN = RED = RESET = CYAN = ""
    else:
        GREEN = RED = RESET = CYAN = ""

    header = ["n", "method", "iters", "f(x*)", "||grad||", "time(s)", "ok", "reason", "status_detail"]
    print("{:>3}  {:>20}  {:>6}  {:>12}  {:>10}  {:>8}  {:>3}  {:>8}  {}".format(*header))

    for r in rows:
        it = "-" if r.get("iters") is None else r["iters"]
        grad_norm = float(r["grad_norm"])
        fval = float(r["fval"])
        df = abs(float(r.get("df", 0.0)))  # if not provided, treat as 0

        # --- strict convergence logic + reason label ---
        if grad_norm <= tol_grad:
            ok = True
            reason = "grad"
            status_detail = f"grad_norm={grad_norm:.2e} <= tol_grad={tol_grad:.1e}"
        elif df <= tol_f:
            ok = True
            reason = "fchg"
            status_detail = f"|df|={df:.2e} <= tol_f={tol_f:.1e}"
        elif (r.get("iters") is not None) and (r["iters"] >= max_iters):
            ok = False
            reason = "max_iter"
            status_detail = (f"reached max_iters={max_iters} "
                             f"(grad_norm={grad_norm:.2e}, |df|={df:.2e})")
        else:
            ok = False
            reason = "?"
            status_detail = (f"no criterion met (grad_norm={grad_norm:.2e}, "
                             f"|df|={df:.2e})")

        ok_symbol = f"{GREEN}✓{RESET}" if ok else f"{RED}×{RESET}"
        reason_str = reason
        # 可选：给 reason 上色（更醒目）
        if reason == "grad":
            reason_str = f"{GREEN}{reason}{RESET}"
        elif reason == "fchg":
            reason_str = f"{CYAN}{reason}{RESET}"
        elif reason == "max_iter":
            reason_str = f"{RED}{reason}{RESET}"

        print("{:>3}  {:>20}  {:>6}  {:>12.4e}  {:>10.2e}  {:>8.3f}  {:>3}  {:>8}  {}".format(
            r["n"], r["method"], it, fval, grad_norm, r["time"], ok_symbol, reason_str, status_detail
        ))