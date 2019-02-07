


import cvxpy as cp

q = 1
W = 1
T = 1
h = 4
x_nu = 7/3

y = cp.Variable(1)

objective = cp.Minimize(q * y)


constraints = [0 == W * y + T * x_nu - h, y >= 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.CVXOPT)

print(y.value)
print(constraints[0].dual_value)
print(constraints[1].dual_value)