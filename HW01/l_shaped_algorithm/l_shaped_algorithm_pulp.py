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
import pulp as lp


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
        
        self.debug = False
        self.verbose = True
        
        


        
        
    def solve(self):
        
        for _ in range(5):
            self.nu += 1 # iterate step counter
            print (f"===== Iteration {self.nu} =====")
            x_i, theta_i = self.step_1()

            
            D_step2, d_step2 = self.step_2()
            if np.any(D_step2) != None:
                # A feasibility cut was made
                continue
            else :
                print ("No step 2 cuts made")
        
        
    def step_1(self):
        ''' 
        Solve the linear program with any constraints imposed by previous
        feasibility and optimality cuts
        '''
        print ()        
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
        result = prob.solve(verbose=self.verbose)
        
        x_solution = np.array(x.value).reshape([x.value.size])
        
        print ("x_nu = ")
        print (x_solution)
        print ("theta_nu = ")
        print (theta.value)
            
        # if there are no optimality cuts, set theta_nu = -inf
        if self.s == 0:
            if x.value.all() == None:
                self.x_nu.append(np.zeros([n]))   
            else:
                self.x_nu.append(x_solution)   
            return self.x_nu[-1], self.theta_nu[-1]
        else:
            self.x_nu.append(x_solution)
            self.theta_nu.append(theta.value)
        
        return self.x_nu[-1], self.theta_nu[-1]
        
        
        
    def step_2(self):
        '''
        Solve LPs for each possible realization of the random variable, and 
        make feasibility cuts as appropriate
        '''
        n = self.W.shape[0]
        m = self.W.shape[1]
        
        print ()
        print (f"Starting Step 2 for iteration {self.nu}")
        
        for k in range(self.K):
            lp_name = f"step2_iteration{self.nu}_k={k}"
            print (lp_name)
            
            prob = lp.LpProblem(lp_name, lp.LpMinimize)
            
            vp = lp.LpVariable.matrix("vp", indexs=list(range(n)), lowBound=0)
            vm = lp.LpVariable.matrix("vm", indexs=list(range(n)), lowBound=0)
            
            y = lp.LpVariable.matrix("y", indexs=list(range(m)), lowBound=0)
            
            e_n = np.ones([n])
            
            # Define the objective function
            prob += lp.lpDot(e_n, vp) + lp.lpDot(e_n, vm), "obj"

            for i in range(n):
                if self.h[0][i] >= 0:
                    prob += ( lp.lpDot(self.W[i], y) + vp[i] - vm[i] <= 
                             self.h[k][i] - self.T[k][i] @ self.x_nu[-1], 
                             "constraint" + str(i) )
                else:
                    prob += ( lp.lpDot(-1*self.W[i], y) + vp[i] - vm[i] >= 
                             -1*self.h[k][i] - self.T[k][i] @ self.x_nu[-1], 
                             "constraint" + str(i) )

            prob.solve()
        
            print("Status:", lp.LpStatus[prob.status])
            print("objective=", lp.value(prob.objective))
            
            if self.debug:
                for v in prob.variables():
                    print(v.name, "=", v.varValue, "\tReduced Cost =", v.dj)

                print("\nSensitivity Analysis\nConstraint\t\t\t\t\t\tShadow Price\tSlack")
                for name, c in prob.constraints.items():
                    print(name, ":", c, "\t", c.pi, "\t\t\t\t\t\t", c.slack)    
    
            
            if lp.value(prob.objective) > 0:
                # Then we need to add a feasibility cut
                
                print ("Feasibility cut needed")
                # Get the dual variables
                sigma = []    
                for name, c in prob.constraints.items():
                    sigma.append(c.pi)
                
                sigma = np.array(sigma)
                print ("dual objective = ", (np.abs(self.h[0]) - self.T[0]@self.x_nu[-1]) @ sigma)
        
                print ("dual variables = ")
                print (sigma)
                
                D = sigma.T @ self.T[k]
                d = sigma.T @ np.abs(self.h[k])
                if d < 0.0:
                    d *= -1
                    D *= -1
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
