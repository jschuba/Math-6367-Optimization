import cvxpy as cp
import numpy as np

vp1 = cp.Variable()
vp2 = cp.Variable()
vp3 = cp.Variable()
vp4 = cp.Variable()
vp5 = cp.Variable()
vp6 = cp.Variable()

vm1 = cp.Variable()
vm2 = cp.Variable()
vm3 = cp.Variable()
vm4 = cp.Variable()
vm5 = cp.Variable()
vm6 = cp.Variable()

y1 = cp.Variable()
y2 = cp.Variable()


objective = cp.Minimize(vp1 + vp2 + vp3 + vp4 + vp5 + vp6
                        + vm1 + vm2 + vm3 + vm4 + vm5 + vm6)

constraints = [vp1 - vm1 + 3*y1 + 2*y2 <= 0, 
               vp2 - vm2 + 2*y1 + 5*y2 <= 0,
               vp3 - vm3 + y1 >= 4.8,
               vp4 - vm4 + y2 >= 6.4,
               vp5 - vm5 + y1 <= 6,
               vp6 - vm6 + y2 <= 8,
               vp1 >= 0,
               vp2 >= 0,
               vp3 >= 0,
               vp4 >= 0,
               vp5 >= 0,
               vp6 >= 0,
               vm1 >= 0,
               vm2 >= 0,
               vm3 >= 0,
               vm4 >= 0,
               vm5 >= 0,
               vm6 >= 0,
               y1 >= 0,
               y2 >= 0
               ]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.CVXOPT, verbose=True)

print(result)

for c in constraints:
    print (c.dual_value)


