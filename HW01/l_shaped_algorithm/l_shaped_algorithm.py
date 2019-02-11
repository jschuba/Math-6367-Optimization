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
import cvxpy as cp
import numpy as np



from sklearn import preprocessing



class L_Shaped_Algorithm():
    """
    
    """
    
    def __init__(self, c, A, b, W, h, T, p, solver='CVX'):
        
        self.c = c
        self.A = A
        self.b = b
        self.W = W
        self.h = h
        self.T = T
        self.p = p
        
        self.solver = solver
        
        # Check that lengths match
        self.K = len(h)
        if self.K != len(T) or self.K != len(p):
            raise ValueError("h, T, and p should be same length")
        
        
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
        
        


        
        
    def solve(self):
        
        for _ in range(2):
            self.nu += 1 # iterate step counter
            print (f"===== Iteration {self.nu} =====")
            x_i, theta_i = self.step_1()

            
            D, d = self.step_2()
            if D.any() != None:
                continue
        
        
    def step_1(self):
        ''' 
        Solve the linear program with any constraints imposed by previous
        feasibility and optimality cuts
        '''
                
        print (f"Starting Step 1 for iteration {self.nu}")
        
        n = len(self.c)         
        x = cp.Variable(n)
        theta = cp.Variable(1)
        
        if self.s == 0:
            objective = cp.Minimize(self.c @ x)
            self.theta_nu.append(-np.inf)
        else:
            objective = cp.Minimize(self.c @ x + theta)
            
        constraints = [x >= 0]
        if self.A is not None:
            constraints.append( self.A @ x == self.b )
                       
        
        for i in range(len(self.D)):
            # add constraints for each feasibility cut
            print ("Adding feasibility cut")
            constraints.append( self.D[i] @ x >= self.d[i])
        for i in range(len(self.E)):
            # add constraints for each optimality cut
            constraints.append( self.E[i] @ x + theta >= self.e[i])
            
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=True)
        
        print ("x_nu = ")
        print (x.value)
        print ("theta_nu = ")
        print (theta.value)
            
        # if there are no optimality cuts, set theta_nu = -inf
        if self.s == 0:
            if x.value.all() == None:
                self.x_nu.append(np.zeros([n]))   
            else:
                self.x_nu.append(np.array(x.value).reshape([x.value.size]))
                print (x.value.shape)
            return self.x_nu[-1], self.theta_nu[-1]
        else:
            self.x_nu.append(np.array(x.value.reshape([x.value.size])))
            self.theta_nu.append(theta.value)
        
        return self.x_nu[-1], self.theta_nu[-1]
        
        
        
    def step_2(self):
        '''
        Solve LPs for each possible realization of the random variable, and 
        make feasibility cuts as appropriate
        '''
        n = self.W.shape[0]
        m = self.W.shape[1]
        
        print (f"Starting Step 2 for iteration {self.nu}")
        
        for k in range(self.K):
            v_plus = cp.Variable(n)
            v_minus = cp.Variable(n)
            y = cp.Variable(m)
            
            print (self.x_nu[-1])
            print (self.h[k])
            
            
            
            objective = cp.Minimize(np.ones(n) @ v_plus
                                    + np.ones(n) @ v_minus)
            constraints = [self.W @ y + v_plus - v_minus == self.h[k] - self.T[k] @ self.x_nu[-1],
                           y >= 0,
                           v_plus >= 0,
                           v_minus >= 0]
            
            prob = cp.Problem(objective, constraints)
            result = prob.solve(verbose=False)
            
            if result > 0:
                # The problem is infeasible, so introduce constraints
                print ()
                print (f"Infeasibility found for realization {k}")
                print (f"The result was {result}")
                print ("v_plus = ")
                print (v_plus.value)
                print ("v_minus = ")
                print (v_minus.value)
                print()
                
                
                sigma = -1 * constraints[0].dual_value
                print ("Dual Value is ")
                print (sigma)
#                for c in constraints:
#                    print (c.dual_value)
                print ( self.W @ y.value + v_plus.value - v_minus.value)
                print ( self.h[k] - self.T[k] @ self.x_nu[-1] )
                print ( (self.h[k] - self.T[k] @ self.x_nu[-1]) @ sigma )
                self.r += 1  # increment counter for feasibility cuts
                D = sigma.T @ self.T[k]
                d = sigma.T @ self.h[k]

                
                print ("Dk = ")
                print (D)
                print ("dk = ")
                print (d)
                self.D.append(D)
                self.d.append(d)
                return D, d
        
        return None, None
        
        
        
    def step_3(self):
        pass
