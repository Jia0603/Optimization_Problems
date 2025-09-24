import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from base import OptimizationProblem
from Optimizers import NewtonOptimizer


def extended_rosenbrock(x, n=10):
    """扩展Rosenbrock函数到n维"""
    x = np.asarray(x, dtype=float)
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
              for i in range(n-1))

def extended_rosenbrock_grad(x, n=10):
    """扩展Rosenbrock函数的梯度"""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    
    for i in range(n-1):
        grad[i] += -400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1 - x[i])
        grad[i+1] += 200.0 * (x[i+1] - x[i]**2)
    
    return grad


def debug_10d_convergence():
    """调试10维问题的收敛情况"""
    n = 10
    prob = OptimizationProblem(
        lambda x: extended_rosenbrock(x, n),
        lambda x: extended_rosenbrock_grad(x, n)
    )
    
    x0 = np.ones(n) * (-1.2)
    x0[1::2] = 1.0
    
    print("10维Rosenbrock函数收敛调试")
    print("=" * 50)
    print(f"初始点: {x0}")
    print(f"初始函数值: {prob.f(x0):.6f}")
    print(f"目标点: {np.ones(n)}")
    print()
    
    methods = ['exact', 'inexact']
    
    for method in methods:
        print(f"\n{method.upper()} 线性搜索:")
        print("-" * 30)
        
        opt = NewtonOptimizer(prob, line_search_type=method, tol=1e-4, max_iter=50)
        x_star = opt.solve(x0.copy())
        
        print(f"最终点: {x_star}")
        print(f"最终函数值: {prob.f(x_star):.6e}")
        print(f"最终梯度范数: {np.linalg.norm(prob.g(x_star)):.6e}")
        print(f"与目标点距离: {np.linalg.norm(x_star - np.ones(n)):.6e}")
        print(f"迭代次数: {opt.iter_info['iter']}")
        
        # 检查是否真的收敛到正确解
        if np.allclose(x_star, np.ones(n), atol=1e-2):
            print("✅ 收敛到正确解")
        else:
            print("❌ 没有收敛到正确解!")
            
        # 显示收敛历史
        print("\n收敛历史 (最后5步):")
        f_history = opt.iter_info['f']
        g_history = opt.iter_info['gradient']
        for i in range(max(0, len(f_history)-5), len(f_history)):
            f_val = f_history[i]
            g_norm = np.linalg.norm(g_history[i])
            print(f"  迭代 {i+1}: f={f_val:.6e}, |g|={g_norm:.6e}")


if __name__ == "__main__":
    debug_10d_convergence()
