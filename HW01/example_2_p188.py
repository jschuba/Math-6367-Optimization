import numpy as np
from l_shaped_algorithm.l_shaped_algorithm import L_Shaped_Algorithm

c = np.array([0])

A_ineq = [1]
b_ineq = [10]

W = np.array([1])

h = []
T = []

q = [[1],[1],[1]]
s = [1,2,4]
p = [1/3,1/3,1/3]

def T_func(x, s):
    if x <= s:
        return np.array([1])
    else:
        return np.array([-1])
        
def h_func(x, s):
    if x <= s:
        return np.array([s])
    else:
        return np.array([-s])        

Solver = L_Shaped_Algorithm(c = c, 
                            A_eq = None, 
                            b_eq = None, 
                            A_ineq = A_ineq, 
                            b_ineq = b_ineq, 
                            W = W, 
                            h_func = h_func, 
                            T_func = T_func, 
                            q = q, 
                            realizations = s, 
                            probabilities = p, 
                            max_iter = 100, 
                            precision=10e-6, 
                            verbose=False, debug=False)
x_opt = Solver.solve()

print (Solver.value)
print (Solver.solution)

