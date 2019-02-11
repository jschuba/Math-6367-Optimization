import cvxpy as cp
import numpy as np

q = np.array([1, 3])
W = np.array([[1, 4], [6,4]])
T = np.array([[1, 0], [0,1]])
h = np.array([4,5])
x_nu = np.array([1, 2])

y = cp.Variable(2)


objective = cp.Minimize(q @ y)

constraints = [ W @ y == h - T @ x_nu, y >= 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.CVXOPT, verbose=True)

print(y.value)
print(constraints[0].dual_value)
print(constraints[1].dual_value)

sigma_dual = constraints[0].dual_value * -1

#%%

sigma = cp.Variable(2)
s = cp.Variable(2)
objective = cp.Maximize( (h - T @ x_nu) @ sigma)
constraints = [W@sigma + s == q, s>=0]
prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.CVXOPT, verbose=True)

print(sigma.value)

print(q@y.value / (h - T @ x_nu))

print( q@ y.value)
print((h - T @ x_nu) @ sigma_dual)


