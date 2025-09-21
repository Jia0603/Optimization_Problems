# Optimization_Problems
Implementation of some classical optimization algorithms.
# 文件结构建议
project02/
    __init__.py
    base.py        # 包含 OptimizationProblem 和 Optimizer 两个基类 （task 1 & 2）
    Otimizers.py   # 已完成经典牛顿法实现（task 3）； 需增加支持精确及非精确线性搜索的牛顿法实现（task 4 & task 6），继承自Optimizer基类
    QNOptimizers.py # 五种QuasiNewtonOptimizer类（task9） (BroydenGood, BroydenBad, SymmetricBroyden, DFP, BFGS)，继承自Optimizer基类
    Implementation.ipynb # demo notebook
    tests/
        __init__.py
        test_rosenbrock.py # task 5 & task 7 (简单验证，可以和task4、task6分别打包做)
        test_chebyquad.py # task 10 & task 11
        benchmark.py  # 与 scipy 对比， task 12
