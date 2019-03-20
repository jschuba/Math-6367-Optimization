

import numpy as np
from l_shaped_algorithm.l_shaped_algorithm import L_Shaped_Algorithm

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
        
def T_func(x, s):
    return np.array([[-1,0],
                     [0,-1],
                     [0,0],
                     [0,0],
                     [0,0],
                     [0,0]])

def h_func(x, s):
    return np.array([0, 0, -0.8*s[0], -0.8*s[1], s[0], s[1]])        

Solver = L_Shaped_Algorithm(c = c, 
                            A_eq = None, 
                            b_eq = None, 
                            A_ineq = None, 
                            b_ineq = None, 
                            W = W, 
                            h_func = h_func, 
                            T_func = T_func, 
                            q = q, 
                            realizations = s, 
                            probabilities = p, 
                            max_iter = 100, 
                            precision=10e-6, 
                            verbose=True, debug=False)
x_opt = Solver.solve()
#
#
#from matplotlib import pyplot as plt
#
#def convert_to_line(a, c):
#    line = lambda x: [(c - a[0]*xi)/a[1] for xi in x]
#    return line
#
#solutions = np.array(Solver.x_nu_list)
#
#scale_max = np.max(solutions)
#
#fig = plt.figure()
#ax = fig.gca()
#
#x = np.linspace(0,scale_max,num=100)
#
#colors = ['green', 'red', 'yellow', 'blue', 'purple']
#
#for k in range(len(Solver.D_list)):
#    a = Solver.D_list[k]
#    c = Solver.d_list[k]
#    line = convert_to_line(a,c)(x)
#    ax.plot(x, line, color = colors[k])
#    ax.fill_between(x, 0, line, facecolor=colors[k], alpha = 0.3, interpolate=True)
#    
#ax.scatter(solutions[:,0], solutions[:,1])
#
#ax.set_xlim(0,scale_max*1.2)
#ax.set_ylim(0,scale_max*1.2)
#
