"""
L-Shaped Algorithm

A method for solving two-stage stochastic linear program with recourse

Implemented for:

Math 6367 Optimization (Prof H.W. Hoppe)
HW01 
Due 20 March 2019

Students:
Jonathan Schuba

"""

import numpy as np
import cvxpy as cp

from sklearn import preprocessing



class L_Shaped_Algorithm():
    """
    
    """
    
    def __init__(self, c, A, b solver='simplex'):
        
        self.c = c
        self.A = A
        self.b = b
        
        self.solver = solver
        
        
        ''' Initialize counters and lists to store computed quantities '''
        self.nu = 0         # iteration counter
        self.r = 0          # counter for feasibility cuts
        self.s = 0          # counter for optimality cuts
        
        ''' Matrices and vectors which will form constraints pertaining to 
        feasibility cuts, ie:
            D[i] @ x >= d[i]  where 1<= i <=r
        '''
        self.D = []         # list of matrices for feasibility cuts
        self.d = []         # list of vectors for feasibility cuts
        
        ''' Matrices and vectors which will form constraints pertaining to 
        optimality cuts, ie:
            E[i] @ x >= e[i]  where 1<= i <=s
        '''
        self.E = []         # list of matrices for optimality cuts
        self.e = []         # list of vectors for optimality cuts
        
        ''' Lists to hold the values obtained in each iteration 
        '''
        self.x_nu = []
        self.theta_nu = []
        
        


        
        
    def solve():
        
        x_i, theta_i = step_1()
        
        
    def step_1():
        ''' 
        Solve the linear program with any constraints imposed by previous
        feasibility and optimality cuts
        '''
        
        # Skip step 1 for first iteration
        if self.nu == 0:
            self.x_nu.append(0)
            self.theta_nu.append(-np.inf)
            return self.x_nu[-1], self.theta_nu[-1]
        
        n = len(self.c)         
        x = cp.Variable(n)
        theta = cp.Variable(1)
        
        objective = cp.Minimize(self.c @ x + theta)
        constraints = [ self.A @ x = self.b,
                       x >= 0]
        
        for i in len(self.D):
            # add constraints for each feasibility cut
            constraints.append( self.D[i] @ x >= self.d[i])
        for i in len(self.E):
            # add constraints for each optimality cut
            constraints.append( self.E[i] @ x + theta >= self.e[i])
            
        prob = cp.Problem(objective, constraints)
        result = prob.solve()            
        
        self.x_nu.append(x.value)
        self.theta_nu.append(theta.value)
        
        return self.x_nu[-1], self.theta_nu[-1]
        
        
        
    def step_2():
        
    def step_3():
        
