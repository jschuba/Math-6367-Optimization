

import numpy as np
from l_shaped_algorithm_pulp_v2 import L_Shaped_Algorithm

c = np.array([3,2])

W = np.array([[3,2,1,0,0,0,0,0], 
              [2,5,0,1,0,0,0,0],
              [-1,0,0,0,1,0,0,0],
              [0,-1,0,0,0,1,0,0],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,0,0,0,1]])

'''
W = np.array([[3,2], 
              [2,5],
              [-1,0],
              [0,-1],
              [1,0],
              [0,1]
              ])
'''
h = []
T = []
p = []
q = []
for s1 in [4,6]:
    for s2 in [8,4]:
        h.append(np.array([0, 0, -0.8*s1, -0.8*s2, s1, s2]))
        T.append(np.array([[-1,0],
                           [0,-1],
                           [0,0],
                           [0,0],
                           [0,0],
                           [0,0]]))
        p.append(1/4)
        q.append(np.array([s1,s2]))
        
        

Solver = L_Shaped_Algorithm(c, None, None, W, h, T, q, p,
                            max_iter = 10, verbose=False, debug=False)

x_opt = Solver.solve()

