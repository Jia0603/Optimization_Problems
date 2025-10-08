# QNOptimizers.py
import numpy as np
from base import Optimizer
from Optimizers import NewtonOptimizer

_EPS = 1e-12


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return (A + A.T) / 2.0


def _is_descent(g: np.ndarray, p: np.ndarray) -> bool:
    # 下降方向要求 p^T g < 0
    return float(g.T @ p) < 0.0


class _QNBase(Optimizer):
    """
    Generic Quasi-Newton base class: provides direction based on H.
    Step size uses the problem's inexact line search (kept as Goldstein here).
    Subclasses need to:
      - initialize_algorithm(x0): init required matrices (H or G, or both)
      - update_algorithm_state(...): update with (s, y)
    """
    def compute_direction(self, x, f_val, g_val):
        # Default: use inverse Hessian approximation H: dir = -H g
        H = self.algorithm_state.get('H', None)
        if H is None:
            # fallback to steepest descent
            direction = -g_val
        else:
            direction = - H @ g_val

        # 下降方向保护：若不是下降方向，退回最速下降
        if not _is_descent(g_val, direction):
            direction = -g_val
        return direction

    def compute_step_size(self, x, direction, f_val, g_val):
        # Goldstein section search (kept as-is by your request)
        newton_like = NewtonOptimizer(self.problem)
        return newton_like.strong_wolfe_line_search(x, direction, f_val, g_val)

    # ---- Utilities for subclasses ----
    def _maybe_scale_H0_with_gamma(self, key_H='H', key_G=None, s=None, y=None):
        """
        首步尺度化 H0：gamma = (s^T y)/(y^T y), H0 = gamma * I
        对 Good Broyden 同步设置 G0 = (1/gamma) * I 以保持互逆尺度。
        仅在满足正性并且尚未尺度化时进行一次。
        """
        if s is None or y is None:
            return
        if self.algorithm_state.get('_H_scaled_once', False):
            return

        ys = float(y @ s)
        yTy = float(y @ y)
        if ys > _EPS and yTy > _EPS:
            gamma = ys / yTy
            n = len(s)
            self.algorithm_state[key_H] = gamma * np.eye(n)

            if key_G is not None:
                # 若需要维护 G（Hessian 近似），设置与 H 互为尺度
                if gamma > _EPS:
                    self.algorithm_state[key_G] = (1.0 / gamma) * np.eye(n)
            self.algorithm_state['_H_scaled_once'] = True


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
        self.algorithm_state['_H_scaled_once'] = False

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        # 首步尺度化 H0 / G0（可大幅减少前几步线搜索挣扎）
        self._maybe_scale_H0_with_gamma(key_H='H', key_G='G', s=s, y=y)

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

        self.algorithm_state['G'] = _symmetrize(G_new)
        self.algorithm_state['H'] = _symmetrize(H_new)


# ---------- 2) Bad Broyden: rank-1 update of H ----------
class BroydenBad(_QNBase):
    """
    Simple Broyden rank-1 update of H = G^{-1} (secant: H_{k+1} y = s)
      H_{k+1} = H_k + ((s - H_k y) y^T) / (y^T y)
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['H'] = np.eye(n)
        self.algorithm_state['_H_scaled_once'] = False

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        # 首步尺度化 H0
        self._maybe_scale_H0_with_gamma(key_H='H', s=s, y=y)

        H = self.algorithm_state['H']
        yTy = float(y @ y)
        if yTy < _EPS:
            return

        Hy = H @ y
        H_new = H + np.outer(s - Hy, y) / yTy
        self.algorithm_state['H'] = _symmetrize(H_new)


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
        self.algorithm_state['_H_scaled_once'] = False

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        # 首步尺度化 H0
        self._maybe_scale_H0_with_gamma(key_H='H', s=s, y=y)

        H = self.algorithm_state['H']
        r = s - H @ y
        denom = float(r @ y)
        # SR1 safeguard: skip update if |denom| is too small
        if abs(denom) > np.sqrt(_EPS):
            H_new = H + np.outer(r, r) / denom
            self.algorithm_state['H'] = _symmetrize(H_new)
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
        self.algorithm_state['_H_scaled_once'] = False

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        # 首步尺度化 H0
        self._maybe_scale_H0_with_gamma(key_H='H', s=s, y=y)

        H = self.algorithm_state['H']
        ys = float(y @ s)
        if ys <= _EPS:
            # 曲率不正，跳过更新（DFP 对噪声较敏感）
            return

        Hy = H @ y
        yHy = float(y @ Hy)
        if yHy <= _EPS:
            return

        term1 = np.outer(s, s) / ys
        term2 = np.outer(Hy, Hy) / yHy
        H_new = H + term1 - term2
        self.algorithm_state['H'] = _symmetrize(H_new)


# ---------- 5) BFGS: rank-2 update of H ----------
class BFGS(_QNBase):
    """
    BFGS inverse update:
      rho = 1/(y^T s)
      H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T

    增强点：
      - 曲率守护：若 y^T s <= 0，跳过或阻尼
      - Powell 阻尼（推荐）：保证 s^T ỹ >= 0.2 * s^T H^{-1} s
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['H'] = np.eye(n)
        self.algorithm_state['_H_scaled_once'] = False

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        # 首步尺度化 H0
        self._maybe_scale_H0_with_gamma(key_H='H', s=s, y=y)

        H = self.algorithm_state['H']

        # ---- Powell 阻尼（可选但很稳）：确保 s^T y 够正 ----
        ys = float(y @ s)
        if ys <= _EPS:
            # 尝试 Powell damping：用 H^{-1}s 做混合
            try:
                Hinvs = np.linalg.solve(H, s)  # 近似 H^{-1} s
                sTHinvs = float(s.T @ Hinvs)
                if sTHinvs > _EPS and ys < 0.2 * sTHinvs:
                    theta = (0.8 * sTHinvs) / (sTHinvs - ys)
                    y = theta * y + (1.0 - theta) * Hinvs
                    ys = float(y @ s)  # 以阻尼后的 ỹ 继续
            except np.linalg.LinAlgError:
                # H 奇异，直接跳过本次更新
                return

        # 若仍不满足正性，跳过更新（保守）
        if ys <= _EPS:
            return

        rho = 1.0 / ys
        I = np.eye(len(s))
        V = I - rho * np.outer(s, y)
        H_new = V @ H @ V.T + rho * np.outer(s, s)
        self.algorithm_state['H'] = _symmetrize(H_new)
