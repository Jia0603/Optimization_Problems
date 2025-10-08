# QNOptimizers.py
import numpy as np
from base import Optimizer
from Optimizers import NewtonOptimizer

_EPS = 1e-12


class _QNBase(Optimizer):
    """
    Generic Quasi-Newton base class: provides direction based on H,
    default step size = 1.0.
    Subclasses only need to:
      - initialize_algorithm(x0): initialize required matrices (H or G, or both)
      - update_algorithm_state(...): update with (s, y)
    """
    def compute_direction(self, x, f_val, g_val):
        # Default: use inverse Hessian approximation H: dir = -H g
        H = self.algorithm_state.get('H', None)
        if H is None:
            # fallback to steepest descent
            return -g_val
        return - H @ g_val

    def compute_step_size(self, x, direction, f_val, g_val):
        # Goldstein section search
        newton_like = NewtonOptimizer(self.problem)
        # return newton_like.inexact_line_search(x, direction, f_val, g_val)
        return newton_like.strong_wolfe_line_search(x, direction, f_val, g_val) # Wolfe



# ---------- 1) Good Broyden: rank-1 update of G, Sherman–Morrison update of H ----------
class BroydenGood(_QNBase):
    """
    Simple Broyden rank-1 update of G, then update H via Sherman–Morrison.
      G_{k+1} = G_k + ((y - G_k s) s^T) / (s^T s)
      If G_{k+1} = G_k + u v^T with u=((y-G_k s)/(s^T s)), v=s,
      then H_{k+1} = H_k - (H_k u v^T H_k) / (1 + v^T H_k u)
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['G'] = np.eye(n)   # Hessian approximation
        self.algorithm_state['H'] = np.eye(n)   # Inverse approximation (for direction)

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        G = self.algorithm_state['G']
        H = self.algorithm_state['H']

        sTs = float(s @ s)
        if sTs < _EPS:
            return  # step too small, skip update

        # --- update G (simple rank-1) ---
        q = y - G @ s
        G_new = G + np.outer(q, s) / sTs

        # --- update H with Sherman–Morrison ---
        u = q / sTs
        v = s
        Hu = H @ u
        vTHu = float(v @ Hu)
        denom = 1.0 + vTHu
        if abs(denom) > _EPS:
            H_new = H - np.outer(Hu, H @ v) / denom
        else:
            # fallback: keep old H if unstable
            H_new = H

        self.algorithm_state['G'] = (G_new + G_new.T) / 2.0
        self.algorithm_state['H'] = (H_new + H_new.T) / 2.0


# ---------- 2) Bad Broyden: rank-1 update of H ----------
class BroydenBad(_QNBase):
    """
    Simple Broyden rank-1 update of H = G^{-1} (secant: H_{k+1} y = s)
      H_{k+1} = H_k + ((s - H_k y) y^T) / (y^T y)
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['H'] = np.eye(n)

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        H = self.algorithm_state['H']
        yTy = float(y @ y)
        if yTy < _EPS:
            return

        Hy = H @ y
        H_new = H + np.outer(s - Hy, y) / yTy
        self.algorithm_state['H'] = (H_new + H_new.T) / 2.0


# ---------- 3) Symmetric Broyden (SR1): symmetric rank-1 update of H ----------
class SymmetricBroyden(_QNBase):
    """
    SR1 (symmetric rank-1) inverse update:
      H_{k+1} = H_k + ((s - H_k y)(s - H_k y)^T) / ((s - H_k y)^T y)
    Note: when denominator is too small or negative, usually skip the update.
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['H'] = np.eye(n)

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        H = self.algorithm_state['H']
        r = s - H @ y
        denom = float(r @ y)
        # SR1 safeguard: skip update if |denom| is too small
        if abs(denom) > np.sqrt(_EPS):
            H_new = H + np.outer(r, r) / denom
            self.algorithm_state['H'] = (H_new + H_new.T) / 2.0
        # else: skip update


# ---------- 4) DFP: rank-2 update of H ----------
class DFP(_QNBase):
    """
    DFP inverse update:
      H_{k+1} = H_k + (ss^T)/(y^T s) - (H_k y y^T H_k)/(y^T H_k y)
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['H'] = np.eye(n)

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        H = self.algorithm_state['H']
        ys = float(y @ s)
        if abs(ys) < _EPS:
            return

        Hy = H @ y
        yHy = float(y @ Hy)
        if abs(yHy) < _EPS:
            return

        term1 = np.outer(s, s) / ys
        term2 = np.outer(Hy, Hy) / yHy
        H_new = H + term1 - term2
        self.algorithm_state['H'] = (H_new + H_new.T) / 2.0


# ---------- 5) BFGS: rank-2 update of H ----------
class BFGS(_QNBase):
    """
    BFGS inverse update:
      rho = 1/(y^T s)
      H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['H'] = np.eye(n)

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        H = self.algorithm_state['H']
        ys = float(y @ s)
        if abs(ys) < _EPS:
            return

        rho = 1.0 / ys
        I = np.eye(len(s))
        V = I - rho * np.outer(s, y)
        H_new = V @ H @ V.T + rho * np.outer(s, s)
        self.algorithm_state['H'] = (H_new + H_new.T) / 2.0
