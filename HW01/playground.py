


import cvxpy as cp

q = 1
W = 1
T = -1
h = -4
x_nu = 10

y = cp.Variable(1)


objective = cp.Minimize(y)

constraints = [ W * y == h - T * x_nu, y >= 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.CVXOPT, verbose=True)

print(y.value)
print(constraints[0].dual_value)
print(constraints[1].dual_value)

#%%

sigma = cp.Variable(1)
s = cp.Variable(1)
objective = cp.Maximize( (h - T * x_nu) * sigma)
constraints = [W*sigma + s == c, s>=0]
prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.CVXOPT, verbose=True)

print(sigma.value)

print(q*y.value / (h - T * x_nu))

#%%
from scipy.optimize import linprog

c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)


res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
               options={"disp": True})
 

print(res)

#%%
import numpy as np
from scipy.optimize import linprog

c = [1]
A_eq = np.array([1]).reshape([1,1])
b = [h-T*x_nu]
x0_bounds = (0, None)



res = linprog(c, A_eq=A_eq, b_eq=b, bounds=(x0_bounds),
               options={"disp": True},
               method = 'interior-point')
 

print(res)
