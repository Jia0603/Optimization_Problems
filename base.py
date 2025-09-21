import numpy as np

class OptimizationProblem:
    '''
    An optimization problem class.
    Inputs : An objective function and its gradient as an option.
    Outputs : f(x), g(x), approx_g(x) if g is not provided.
    '''
    def __init__(self, function, gradient = None):

        self.function = function
        self.gradient = gradient

    def f(self, x):

        return self.function(x) # Seemingly redundant, but keep it in case of further needs
    
    def g(self, x, eps=1e-6):

        if not self.gradient:
            return self.approx_g(x, eps)
        else:
            return self.gradient(x)
        
    def approx_g(self, x, eps):

        grad = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = 1
            grad[i] = (self.f(x + eps*e) - self.f(x - eps*e)) / (2 * eps)
        return grad

class Optimizer:
    '''
    A general optimization method class.
    Inputs : A problem object; Initialization x0.
    Outputs : Optimization x*.
    '''
    def __init__(self, problem, tol=1e-6, max_iter=1000):

        self.problem = problem
        self.tol = tol
        self.max_iter = max_iter
        self.iter_info = {'x': [], 'f' : [], 'gradient' : [], 'iter' : 0}
        
        self.algorithm_state = {} # save states for Quasi-Newton methods
    def solve(self, x0 : np.ndarray) -> np.ndarray:
        self.iter_info = {'x': [], 'f' : [], 'gradient' : [], 'iter' : 0}
        self.algorithm_state = {}

        x = x0.copy()
        self.initialize_algorithm(x0)

        for i in range(self.max_iter):
            f_val = self.problem.f(x)
            g_val = self.problem.g(x)

            self.iter_info['x'].append(x.copy())
            self.iter_info['f'].append(f_val)
            self.iter_info['gradient'].append(g_val.copy())
            self.iter_info['iter'] = i + 1

            if np.linalg.norm(g_val) < self.tol:
                print(f"Iteration {i+1}: Problem converged!")
                break

            dir = self.compute_direction(x, f_val, g_val)
            alpha = self.compute_step_size(x, dir, f_val, g_val)

            x_old = x.copy()
            g_old = g_val.copy() # save past states for Quasi-Newton methods

            x += alpha * dir

            self.update_algorithm_state(x_old, x, g_old, dir, alpha)

        return x
    
    # All the subclass need to adress compute_direction() function.
    def compute_direction(self, x, f_val, g_val):
        return - g_val 

    # All the subclass need to adress compute_step_size() function.
    def compute_step_size(self, x, direction, f_val, g_val):
        return 0.01
    # Quasi-Newton methods need to adress update_algorithm_state() function.
    def update_algorithm_state(self, x_old, x_new, g_old, dir, alpha):
        pass
    # Quasi-Newton methods need to adress initialize_algorithm() function.
    def initialize_algorithm(self, x0):
        pass






