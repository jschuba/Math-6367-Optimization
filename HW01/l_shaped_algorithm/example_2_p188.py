

import numpy as np
#from l_shaped_algorithm_pulp_v3 import L_Shaped_Algorithm
from l_shaped_algorithm_cvx_v2 import L_Shaped_Algorithm

c = np.array([0])

A_ineq = [1]
b_ineq = [10]

W = np.array([1])

h = []
T = []

q = [[1],[1],[1]]
s = [1,2,4]
p = [1/3,1/3,1/3]

def T_driver(x, s):
    if x <= s:
        return np.array([1])
    else:
        return np.array([-1])
        
def h_driver(x, s):
    if x <= s:
        return np.array([s])
    else:
        return np.array([-s])        



Solver = L_Shaped_Algorithm(c, None, None, A_ineq, b_ineq, W, h_driver, T_driver, q, s, p,
                            max_iter = 10, verbose=True, debug=False)

x_opt = Solver.solve()

