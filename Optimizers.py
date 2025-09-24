import numpy as np
from base import Optimizer
from scipy.optimize import minimize_scalar

class NewtonOptimizer(Optimizer):

    def __init__(self, problem, line_search_type='fixed', tol=1e-6, max_iter=1000, eps=1e-6):

        super().__init__(problem, tol, max_iter)
        self.eps = eps
        self.line_search_type = line_search_type.lower()
    
    def compute_direction(self, x, f_val, g_val):

        n = len(x)
        G = np.zeros((n, n))
        eps = self.eps

        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0

                f_pp = self.problem.f(x + eps*e_i + eps*e_j)
                f_pm = self.problem.f(x + eps*e_i - eps*e_j)
                f_mp = self.problem.f(x - eps*e_i + eps*e_j)
                f_mm = self.problem.f(x - eps*e_i - eps*e_j)

                G[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
        
        H = (G + G.T) / 2
        try:
            dir = np.linalg.solve(H, - g_val)
        except np.linalg.LinAlgError:
            print("Hassian is not invertable! Set -grad as direction.")
            dir = - g_val

        return dir

    def compute_step_size(self, x, dir, f_val, g_val):
        # 默认配置固定步长1.0，后面实现精确/非精确线性搜索时，在下面定义函数，在这里用if调用
        if self.line_search_type == 'fixed':
            return 1.0
        elif self.line_search_type == 'exact':
            return self.exact_line_search(x, dir, f_val, g_val)
        else:
            return 1.0

    def exact_line_search(self, x, direction, f_val, g_val):
        # ϕ(α) = f(x + α d)
        def phi(alpha):
            return self.problem.f(x + alpha * direction)

        # 使用 Brent 方法做一维最小化；先给出一个常见的 bracket
        # 若 bracket 不理想，minimize_scalar 会自行搜索；此处设置 bounds 以加速常见情形
        try:
            res = minimize_scalar(phi, bracket=(0.0, 1.0), method='Brent')
        except Exception:
            res = minimize_scalar(phi, method='Brent')

        if hasattr(res, 'x') and np.isfinite(res.x):
            return float(res.x)
        else:
            return 1.0
