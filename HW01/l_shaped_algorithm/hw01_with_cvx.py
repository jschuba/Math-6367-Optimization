

import numpy as np
from l_shaped_algorithm_cvx import L_Shaped_Algorithm

c = np.array([3,2])

W = np.array([[3,2,1,0,0,0,0,0], 
              [2,5,0,1,0,0,0,0],
              [-1,0,0,0,1,0,0,0],
              [0,-1,0,0,0,1,0,0],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,0,0,0,1]])


p = []  # probability for each realization of the random variable
q = []  # vector q for each realization
s = []  # random variable values for each realization
for s1 in [6,4]:
    for s2 in [8,4]:
        p.append(1/4)
        q.append(np.array([-15,-12]))
        s.append(np.array([s1,s2]))
        
def T_driver(x, s):
    return np.array([[-1,0],
                     [0,-1],
                     [0,0],
                     [0,0],
                     [0,0],
                     [0,0]])

def h_driver(x, s):
    return np.array([0, 0, -0.8*s[0], -0.8*s[1], s[0], s[1]])        

Solver = L_Shaped_Algorithm(c = c, 
                            A_eq = None, 
                            b_eq = None, 
                            A_ineq = None, 
                            b_ineq = None, 
                            W = W, 
                            h_driver = h_driver, 
                            T_driver = T_driver, 
                            q = q, 
                            realizations = s, 
                            probabilities = p, 
                            max_iter = 100, 
                            precision=10e-6, 
                            verbose=False, debug=False)
x_opt = Solver.solve()

