#!/usr/bin/env python
# Test for output of dual variables

# Import PuLP modeler functions
from pulp import *
import numpy as np


W = np.array([[3,2,1,0,0,0,0,0], 
              [2,5,0,1,0,0,0,0],
              [1,0,0,0,1,0,0,0],
              [-1,0,0,0,0,1,0,0],
              [0,1,0,0,0,0,1,0],
              [0,-1,0,0,0,0,0,1]])

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

x = np.array([[0],[0]])
x = x.reshape([x.size])

# A new LP problem
prob = LpProblem("p192", LpMinimize)

vp = LpVariable.matrix("vp", list(range(n)), 0)
vm = LpVariable.matrix("vm", list(range(n)), 0)

y = LpVariable.matrix("y", list(range(m)), 0)

e_n = np.ones([n])


prob += lpDot(e_n, vp) + lpDot(e_n, vm), "obj"

for i in range(W.shape[0]):
    prob += lpDot(W[i], y) + vp[i] - vm[i] == h[0][i] - T[0][i] @ x, "c" + str(i)


prob.writeLP("p192")

prob.solve()

print("Status:", LpStatus[prob.status])

for v in prob.variables():
	print(v.name, "=", v.varValue, "\tReduced Cost =", v.dj)

print("objective=", value(prob.objective))

print("\nSensitivity Analysis\nConstraint\t\t\t\t\t\tShadow Price\tSlack")
for name, c in list(prob.constraints.items()):
    print(name, ":", c, "\t", c.pi, "\t\t\t\t\t\t", c.slack)
    
    
sigma = []    
for name, c in prob.constraints.items():
    sigma.append(c.pi)
    
sigma = np.array(sigma)    
print ((h[0] - T[0]@x) @ sigma)



D = sigma @ T[0]
d = sigma @ h[0]

print ("D = ")
print (D)
print ("d = ")
print (d)
    
    #%%
    

W = np.array([[3,2], 
              [2,5],
              [1,0],
              [-1,0],
              [0,1],
              [0,-1]])

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

x = np.array([[0],[0]])
x = x.reshape([x.size])

# A new LP problem
prob = LpProblem("p192", LpMinimize)

vp = LpVariable.matrix("vp", list(range(n)), 0)
vm = LpVariable.matrix("vm", list(range(n)), 0)

y = LpVariable.matrix("y", list(range(m)), 0)

e_n = np.ones([n])


prob += lpDot(e_n, vp) + lpDot(e_n, vm), "obj"

for i in range(W.shape[0]):
    if h[0][i] >= 0:
        prob += lpDot(W[i], y) + vp[i] - vm[i] <= h[0][i] - T[0][i] @ x, "c" + str(i)
    else:
        prob += lpDot(-1*W[i], y) + vp[i] - vm[i] >= -1*h[0][i] - T[0][i] @ x, "c" + str(i)


prob.writeLP("p192")

prob.solve()

print("Status:", LpStatus[prob.status])

for v in prob.variables():
	print(v.name, "=", v.varValue, "\tReduced Cost =", v.dj)

print("objective=", value(prob.objective))

print("\nSensitivity Analysis\nConstraint\t\t\t\t\t\tShadow Price\tSlack")
for name, c in prob.constraints.items():
    print(name, ":", c, "\t", c.pi, "\t\t\t\t\t\t", c.slack)    
    
sigma = []    
for name, c in prob.constraints.items():
    sigma.append(c.pi)
    
sigma = np.array(sigma)    
print ((h[0] - T[0]@x) @ sigma)



D = sigma @ T[0]
d = sigma @ h[0]

print ("D = ")
print (D)
print ("d = ")
print (d)
    