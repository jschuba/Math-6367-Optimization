import cvxpy as cp
import numpy as np
from l_shaped_algorithm import L_Shaped_Algorithm

c = np.array([3,2])

W = np.array([[3,2,1,0,0,0,0,0], 
              [2,5,0,1,0,0,0,0],
              [1,0,0,0,1,0,0,0],
              [-1,0,0,0,0,1,0,0],
              [0,1,0,0,0,0,1,0],
              [0,-1,0,0,0,0,0,1]])

'''
W = np.array([[3,2], 
              [2,5],
              [1,0],
              [-1,0],
              [0,1],
              [0,-1]])
'''

h = []
T = []
p = []
for s1 in [6,4]:
    for s2 in [8,4]:
        h.append(np.array([0, 0, s1, -0.8*s1, s2, -0.8*s2]))
        T.append(np.array([[1,0],
                           [0,1],
                           [0,0],
                           [0,0],
                           [0,0],
                           [0,0]]))
        p.append(1/4)
        
        
n = W.shape[0]
m = W.shape[1]     

v_plus = cp.Variable(n)
v_minus = cp.Variable(n)
y = cp.Variable(m)

x = np.array([[0],[0]])
x = x.reshape([x.size])
k = 0

objective = cp.Minimize(cp.sum_entries(v_plus) 
                        + cp.sum_entries(v_minus))

constraints = [W @ y + v_plus - v_minus <= h[k] - T[k] @ x,
               y >= 0,
               v_plus >= 0,
               v_minus >= 0]   

prob = cp.Problem(objective, constraints)
result = prob.solve(verbose=True)

print (result)


nu1 = cp.Variable(n)
lm1 = cp.Variable(m)
lm2 = cp.Variable(n)
lm3 = cp.Variable(n)
y = cp.Variable(m)


objective_dual = cp.Maximize(nu1.T @ ( h[0] - T[0]@x))
constraints_dual = [np.ones(n) + nu1 + lm2 == 0,
                    np.ones(n) - nu1 + lm2 == 0,
                    lm1 >= 0,
                    lm2 >= 0,
                    lm3 >= 0,
                    y >= 0          ]

prob_dual = cp.Problem(objective_dual, constraints_dual)
result_dual = prob_dual.solve(solver = "SCS" ,verbose=True)

print (result_dual)

print (nu1.value)
print (y.value)

for c in constraints_dual:
    print (c.dual_value)
