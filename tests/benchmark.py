"""
Task 12 utilities: rebuild H_k offline and evaluate BFGS inverse-Hessian quality.
This module avoids plotting so it can be imported from notebooks and tests.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional

from problems.chebyquad_adapter import make_chebyquad_problem

# ---------- Numeric Hessian (centered differences) ----------

def numeric_hessian(f: Callable[[np.ndarray], float],
                    x: np.ndarray,
                    eps: float = 5e-6,
                    clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    x = x.astype(float, copy=True)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    def f_eval(z: np.ndarray) -> float:
        if clip_bounds is not None:
            lb, ub = clip_bounds
            z = np.clip(z, lb, ub)
        return float(f(z))

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        for j in range(i, n):
            ej = np.zeros(n); ej[j] = 1.0
            fpp = f_eval(x + eps*ei + eps*ej)
            fpm = f_eval(x + eps*ei - eps*ej)
            fmp = f_eval(x - eps*ei + eps*ej)
            fmm = f_eval(x - eps*ei - eps*ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[j, i] = H[i, j]
    return 0.5 * (H + H.T)  # enforce symmetry

# ---------- Offline BFGS rebuild from iterate sequence ----------

def rebuild_bfgs_H_sequence(f: Callable[[np.ndarray], float],
                            g: Callable[[np.ndarray], np.ndarray],
                            xs: np.ndarray,
                            use_scaled_H0: bool = True,
                            damped: bool = False) -> List[np.ndarray]:
    """
    Given iterates xs[k], rebuild standard (or damped) BFGS inverse-Hessian H_k.
    Independent of optimizer internals; uses xs and gradients along xs only.
    """
    xs = np.asarray(xs)
    n = xs.shape[1]
    Hs: List[np.ndarray] = []
    H = np.eye(n)
    Hs.append(H.copy())

    # gradients along the path
    gs = [g(x) for x in xs]

    # Optional H0 scaling using first step
    if use_scaled_H0 and len(xs) >= 2:
        s0 = xs[1] - xs[0]
        y0 = gs[1] - gs[0]
        yy = float(y0 @ y0)
        ys = float(y0 @ s0)
        if np.isfinite(yy) and yy > 1e-16 and np.isfinite(ys) and ys > 1e-12:
            rho0 = ys / yy
            if rho0 > 1e-12:
                H = rho0 * np.eye(n)
                Hs[0] = H.copy()

    for k in range(len(xs) - 1):
        s = xs[k+1] - xs[k]
        y = gs[k+1] - gs[k]
        ys = float(y @ s)
        if not np.isfinite(ys) or ys <= 1e-12:
            Hs.append(H.copy())
            continue

        if damped:
            sHs = float(s @ (H @ s))
            sHs = max(sHs, 1e-16)
            theta = 1.0
            if ys < 0.2 * sHs:
                theta = (0.8 * sHs) / (sHs - ys + 1e-16)
            y_bar = theta * y + (1.0 - theta) * (H @ s)
            rho = float(s @ y_bar)
            if not np.isfinite(rho) or abs(rho) < 1e-12:
                Hs.append(H.copy()); continue
            V = np.eye(n) - (np.outer(s, y_bar) / rho)
            H = V @ H @ V.T + (np.outer(s, s) / rho)
        else:
            rho = 1.0 / ys
            V = np.eye(n) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        H = 0.5 * (H + H.T)
        Hs.append(H.copy())
    return Hs

# ---------- Collect xs from optimizer (no source-code changes) ----------

class _LoggingProblem:
    """
    Minimal wrapper to record gradient evaluation points.
    If optimizer already exposes iterates (opt.iter_info['x']), prefer that.
    """
    def __init__(self, base):
        self.base = base
        self.visited: List[np.ndarray] = []

    def f(self, x: np.ndarray) -> float:
        return self.base.f(x)

    def g(self, x: np.ndarray) -> np.ndarray:
        self.visited.append(x.copy())
        return self.base.g(x)

def run_and_get_xs(problem, optimizer_cls, x0, tol=1e-8, max_iter=1000):
    """
    Try to fetch xs from opt.iter_info['x']; if not present, wrap and log.
    """
    # Attempt 1: direct run and read iter_info
    try:
        opt = optimizer_cls(problem, tol=tol, max_iter=max_iter)
        x_star = opt.solve(x0)
        if hasattr(opt, "iter_info") and isinstance(opt.iter_info, dict) and "x" in opt.iter_info:
            xs = np.array(opt.iter_info["x"])
            return xs, x_star
    except TypeError:
        # Constructor signature differs: try wrapper path
        pass

    # Attempt 2: wrapper to log gradient points
    lp = _LoggingProblem(problem)
    opt = optimizer_cls(lp, tol=tol, max_iter=max_iter)
    x_star = opt.solve(x0)
    xs_raw = np.array(lp.visited, dtype=float)
    # De-duplicate consecutively equal points
    xs = [xs_raw[0]]
    for x in xs_raw[1:]:
        if np.linalg.norm(x - xs[-1]) > 1e-12:
            xs.append(x)
    xs = np.array(xs)
    return xs, x_star

# ---------- Public API for Task 12 ----------

def evaluate_bfgs_inverse_quality(n: int,
                                  optimizer_cls,
                                  tol: float = 1e-8,
                                  max_iter: int = 1000,
                                  eps_hess: float = 5e-6,
                                  use_scaled_H0: bool = True,
                                  damped: bool = False) -> Dict[str, Any]:
    """
    Run optimizer on chebyquad(n), collect xs, rebuild H_k offline,
    and compute error curves measuring H_k ≈ G(x_k)^{-1}.
    Returns dict with arrays and summary metrics.
    """
    problem, x0 = make_chebyquad_problem(n, use_grad=True)
    xs, x_star = run_and_get_xs(problem, optimizer_cls, x0, tol=tol, max_iter=max_iter)

    # Rebuild H_k
    Hs = rebuild_bfgs_H_sequence(problem.f, problem.g, xs,
                                 use_scaled_H0=use_scaled_H0, damped=damped)

    errs_invF: List[float] = []
    errs_prodI: List[float] = []
    lb = np.zeros(xs.shape[1]); ub = np.ones(xs.shape[1])
    clip = (lb, ub)

    for xk, Hk in zip(xs, Hs):
        Gk = numeric_hessian(problem.f, xk, eps=eps_hess, clip_bounds=clip)
        Ginv = np.linalg.pinv(Gk)
        e1 = np.linalg.norm(Hk - Ginv, 'fro') / max(1.0, np.linalg.norm(Ginv, 'fro'))
        e2 = np.linalg.norm(Hk @ Gk - np.eye(Gk.shape[0]), 'fro')
        errs_invF.append(float(e1)); errs_prodI.append(float(e2))

    return {
        "n": n,
        "xs": xs,
        "Hs": Hs,
        "errs_invF": np.asarray(errs_invF),
        "errs_prodI": np.asarray(errs_prodI),
        "final_rel_inv_err": float(errs_invF[-1]),
        "final_prodI_err": float(errs_prodI[-1]),
        "x_star": x_star,
    }

# ---------- CLI quick run ----------

if __name__ == "__main__":
    from QNOptimizers import BFGS
    res = evaluate_bfgs_inverse_quality(n=8, optimizer_cls=BFGS, damped=False, use_scaled_H0=True)
    print(f"n={res['n']}, steps={len(res['xs'])-1}")
    print(f"final rel inverse error: {res['final_rel_inv_err']:.3e}")
    print(f"final product-to-I error: {res['final_prodI_err']:.3e}")