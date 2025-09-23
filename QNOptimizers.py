# QNOptimizers.py
import numpy as np
from base import Optimizer

_EPS = 1e-12


class _QNBase(Optimizer):
    """
    通用拟牛顿基类：提供基于 H 的方向，步长默认 1.0。
    子类只需：
      - 在 initialize_algorithm(x0) 里初始化需要的矩阵（H 或 G，或两者）
      - 在 update_algorithm_state(...) 里用 (s, y) 做相应更新
    """
    def compute_direction(self, x, f_val, g_val):
        # 缺省使用逆 Hessian 近似 H：dir = -H g
        H = self.algorithm_state.get('H', None)
        if H is None:
            # 退化为最速下降
            return -g_val
        return - H @ g_val

    def compute_step_size(self, x, direction, f_val, g_val):
        # 与 Newton 当前实现一致，先用固定步长；后续可接线搜索
        return 1.0

    @staticmethod
    def _symeig_guard(M):
        # 可选：确保数值稳定（这里不做矫正，只保证对称写法一致）
        return (M + M.T) / 2.0


# ---------- 1) Good Broyden：对 G 做简单秩-1更新，并用 Sherman–Morrison 更新 H ----------
class BroydenGood(_QNBase):
    """
    Simple Broyden rank-1 update of G, then update H via Sherman–Morrison.
      G_{k+1} = G_k + ((y - G_k s) s^T) / (s^T s)
      If G_{k+1} = G_k + u v^T with u=((y-G_k s)/(s^T s)), v=s,
      then H_{k+1} = H_k - (H_k u v^T H_k) / (1 + v^T H_k u)
    """
    def initialize_algorithm(self, x0):
        n = len(x0)
        self.algorithm_state['G'] = np.eye(n)   # Hessian approx
        self.algorithm_state['H'] = np.eye(n)   # Inverse approx (for direction)

    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        prob = self.problem
        s = x_new - x_old
        g_new = prob.g(x_new)
        y = g_new - g_old

        G = self.algorithm_state['G']
        H = self.algorithm_state['H']

        sTs = float(s @ s)
        if sTs < _EPS:
            return  # 步长太小，跳过更新

        # --- 更新 G（simple rank-1）---
        q = y - G @ s
        G_new = G + np.outer(q, s) / sTs

        # --- 用 SM 更新 H ---
        u = q / sTs
        v = s
        Hu = H @ u
        vTHu = float(v @ Hu)
        denom = 1.0 + vTHu
        if abs(denom) > _EPS:
            H_new = H - np.outer(Hu, H @ v) / denom
        else:
            # 退化：不稳定时保持原值（也可回退到求逆 G_new）
            H_new = H

        self.algorithm_state['G'] = self._symeig_guard(G_new)
        self.algorithm_state['H'] = self._symeig_guard(H_new)


# ---------- 2) Bad Broyden：对 H 做简单秩-1更新 ----------
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
        self.algorithm_state['H'] = self._symeig_guard(H_new)


# ---------- 3) Symmetric Broyden（SR1）对 H 的对称秩-1更新 ----------
class SymmetricBroyden(_QNBase):
    """
    SR1 (symmetric rank-1) inverse update:
      H_{k+1} = H_k + ((s - H_k y)(s - H_k y)^T) / ((s - H_k y)^T y)
    注意：当分母很小或为负时常跳过更新（数值稳健做法）。
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
        # SR1 常规滤波：|denom| 太小就跳过（避免数值爆炸）
        if abs(denom) > np.sqrt(_EPS):
            H_new = H + np.outer(r, r) / denom
            self.algorithm_state['H'] = self._symeig_guard(H_new)
        # else: skip update


# ---------- 4) DFP（秩-2）对 H 的更新 ----------
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
        self.algorithm_state['H'] = self._symeig_guard(H_new)


# ---------- 5) BFGS（秩-2）对 H 的更新 ----------
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
        self.algorithm_state['H'] = self._symeig_guard(H_new)
