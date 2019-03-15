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
import pulp as lp


class L_Shaped_Algorithm():
    """
    
    """
    
    def __init__(self, c, A_eq, b_eq, A_ineq, b_ineq, W, h_driver, T_driver, q, 
                 realizations, probabilities, 
                 max_iter = 100, precision=10e-4, 
                 verbose=False, debug=False):
        
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.W = W
        self.h_driver = h_driver
        self.T_driver= T_driver
        self.q = q
        self.realizations = realizations
        self.p = probabilities
        
        self.max_iter = max_iter
        self.precision = precision
        
        self.debug = debug
        self.verbose = verbose        
        
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=abs(int(np.log10(self.precision))))
        
        '''
        Check that lengths match
        '''
        self.K = len(q)
        if self.K != len(self.p):
            raise ValueError("q and p should be same length")
        
        
        ''' 
        Initialize counters and lists to store computed quantities 
        '''
        self.nu = 0         # iteration counter
        self.r = 0          # counter for feasibility cuts
        self.s = 0          # counter for optimality cuts
        
        ''' 
        Matrices and vectors which will form constraints pertaining to 
        feasibility cuts, ie:
            D[i] @ x >= d[i]  where 1 <= i <= r
        '''
        self.D = []         # list of matrices for feasibility cuts
        self.d = []         # list of vectors for feasibility cuts
        
        ''' 
        Matrices and vectors which will form constraints pertaining to 
        optimality cuts, ie:
            E[i] @ x >= e[i]  where 1 <= i <= s
        '''
        self.E = []         # list of matrices for optimality cuts
        self.e = []         # list of vectors for optimality cuts
        
        ''' 
        Lists to hold the values obtained in each iteration 
        '''
        self.x_nu = []
        self.theta_nu = []
        self.objective_value = []
        
        
    def solve(self):
        
        for _ in range(self.max_iter):
            self.nu += 1 # iterate step counter
            print ()
            print ( "===========================================")
            print (f"=============== Iteration {self.nu} ===============")
            print ( "===========================================")
            
            _ = self.step_1()

            cut_made = self.step_2()
            
            if cut_made == 1:
                # A feasibility cut was made
                # Go back to step 1
                continue
            else:
                print ("No feasibility cuts needed")
                
            cut_made = self.step_3()
            if cut_made == 0:
                # optimal solution found
                round_precision = abs(int(np.log10(self.precision)))
                self.value = np.round(self.objective_value[-1], round_precision)
                self.solution = np.round(self.x_nu[-1], round_precision)
                print ()
                print ("Optimal Solution Found")
                print ()
                print ("Objective Value  = ", self.value)
                print ("Optimal Solution = ", self.solution)          
                return self.solution
        
        print ()
        print (f"Maximum iterations ({self.max_iter}) reached, and no ", 
               "optimal solution found")
        print ("Try increasing max_iter or decreasing precision")
        return None
        
        
    def step_1(self):
        ''' 
        Solve the linear program with any constraints imposed by previous
        feasibility and optimality cuts
        '''
      
        print (f"----------------- Step 1 -----------------")
        
        n = len(self.c)         
        
        lp_name = f"step1_iteration{self.nu}"
        prob = lp.LpProblem(lp_name, lp.LpMinimize)
        
        x = lp.LpVariable.matrix("x", indexs=list(range(n)), lowBound=0)
        theta = lp.LpVariable("theta")
        
        if self.s == 0:
            # There are no optimality cuts, so set theta to -inf
            prob += lp.lpDot(self.c, x)
            theta_solution = -np.inf
        else:
            prob += lp.lpDot(self.c, x) + theta
            
        
        if self.A_eq is not None:
            # We must append the equality constraints on x
            for i in range(len(self.A_eq)):
                # Add a scalar constraint for each row of A
                prob += lp.lpDot(self.A_eq[i], x) == self.b_eq[i]
        if self.A_ineq is not None:
            # We must append the inequality constraints on x
            for i in range(len(self.A_ineq)):
                # Add a scalar constraint for each row of A
                prob += lp.lpDot(self.A_ineq[i], x) <= self.b_ineq[i]
                       
            
        for r in range(len(self.D)):
            # add constraints for each feasibility cut
            prob += lp.lpDot(self.D[r], x) >= self.d[r]
        for s in range(len(self.E)):
            # add constraints for each optimality cut
            prob += lp.lpDot(self.E[s], x) + theta >= self.e[s]
            
        prob.solve()
        result = lp.value(prob.objective)
        
        if self.debug:
            for v in prob.variables():
                print(v.name, "=", v.varValue, "\tReduced Cost =", v.dj)

            print("\nSensitivity Analysis\nConstraint\t\t\t\t\t\tShadow Price\tSlack")
            for name, c in prob.constraints.items():
                print(name, ":", c, "\t", c.pi, "\t\t\t\t\t\t", c.slack)    
    
        if result is None and self.nu == 1:
            self.objective_value.append(0)
            self.x_nu.append(np.zeros(self.c.shape))
            self.theta_nu.append(-np.inf)
            return 1
        
        x_solution = []
        for v in prob.variables():
            if "x" in v.name:
                x_solution.append(v.varValue)
            if "theta" in v.name:
                theta_solution = v.varValue
        
        x_solution = np.array(x_solution)
        
        print ("objective value = ", result)
        print ("x_nu            = ", x_solution)
        print ("theta_nu        = ", theta_solution)
            
        self.objective_value.append(result)
        self.x_nu.append(x_solution)
        self.theta_nu.append(theta_solution)
        
        return 1
        
        
        
    def step_2(self):
        '''
        Solve LPs for each possible realization of the random variables, and 
        make feasibility cuts as appropriate
        '''
        n = self.W.shape[0]
        if len(self.W.shape) > 1:
            m = self.W.shape[1]
        else:
            m = 1
        
        print ()
        print (f"----------------- Step 2 -----------------")
        
        for k in range(self.K):
            lp_name = f"step2_iteration{self.nu}_k={k}"
            
            prob = lp.LpProblem(lp_name, lp.LpMinimize)
            
            vp = lp.LpVariable.matrix("vp", indexs=list(range(n)), lowBound=0)
            vm = lp.LpVariable.matrix("vm", indexs=list(range(n)), lowBound=0)
            y = lp.LpVariable.matrix("y", indexs=list(range(m)), lowBound=0)
            
            e_n = np.ones([n])
            
            # Define the objective function
            prob += lp.lpDot(e_n, vp) + lp.lpDot(e_n, vm), "obj"
            
            # We use the user-specified driver functions to get the correct
            # matrix T and h for this particular realization of the random
            # variables
            T = self.T_driver(self.x_nu[-1], self.realizations[k])
            h = self.h_driver(self.x_nu[-1], self.realizations[k])

            # Define constraints of the type Wy + vp - vm == h[k] - T[k]x
            for i in range(n):
                prob += ( lp.lpDot(self.W[i], y) + vp[i] - vm[i] == 
                             h[i] - lp.lpDot(T[i], self.x_nu[-1]), 
                             "constraint" + str(i) )

            prob.solve()
        
            
            if self.debug:
                for v in prob.variables():
                    print(v.name, "=", v.varValue, "\tReduced Cost =", v.dj)

                print("\nSensitivity Analysis\nConstraint\tShadow Price\tSlack")
                for name, c in prob.constraints.items():
                    print(name, ":", c, "\t", c.pi, "\t", c.slack)    
    
            
            if lp.value(prob.objective) > 0:
                # Then we need to add a feasibility cut
                self.r += 1
                
                # Get the dual variables
                sigma = []    
                for name, c in prob.constraints.items():
                    sigma.append(c.pi)
                sigma = np.array(sigma)
                
                print ("Feasibility cut identified")
                print (lp_name)
                print ("objective      = ", lp.value(prob.objective))
                print ("dual objective = ", (h - T @ self.x_nu[-1]) @ sigma)
                print ("dual variables = ", sigma)
                
                D = sigma.T @ T
                d = sigma.T @ h
                print ("Dk = ", D)
                print ("dk = ", d)
                self.D.append(D)
                self.d.append(d)
                return 1 # cut was made
            
        # If we get through all realizations of the random variables, and no
        # infeasibilities were identified, then return 0
        return 0 # cut was not needed
        
        
        
    def step_3(self):
        '''
        Solve LPs for each possible realization of the random variable, and 
        make optimality cuts as appropriate
        '''
        
        n = self.W.shape[0]
        if len(self.W.shape) > 1:
            m = self.W.shape[1]
        else:
            m = 1
            
        print ()
        print (f"----------------- Step 3 -----------------")
        
        # Setup the variables E and e
        E = np.zeros(len(self.x_nu[-1]))
        e = 0
        
        for k in range(self.K):
            lp_name = f"step3_iteration{self.nu}_k={k}"
            
            prob = lp.LpProblem(lp_name, lp.LpMinimize)
            
            y = lp.LpVariable.matrix("y", indexs=list(range(m)), lowBound=0)
            
            # Define the objective function
            prob += lp.lpDot(self.q[k], y), "obj"
            
            # We use the user-specified driver functions to get the correct
            # matrix T and h for this particular realization of the random
            # variables
            T = self.T_driver(self.x_nu[-1], self.realizations[k])
            h = self.h_driver(self.x_nu[-1], self.realizations[k])

            for i in range(n):
                prob += ( lp.lpDot(self.W[i], y) == 
                             h[i] - lp.lpDot(T[i], self.x_nu[-1]), 
                             "constraint" + str(i) )

            prob.solve()
        
            
            if self.debug:
                print (lp_name)
                print (prob.objective)
                for v in prob.variables():
                    print(v.name, "=", v.varValue, "\tReduced Cost =", v.dj)

                print("\nSensitivity Analysis\nConstraint\t\t\t\t\t\tShadow Price\tSlack")
                for name, c in prob.constraints.items():
                    print(name, ":", c, "\t", c.pi, "\t\t\t\t\t\t", c.slack)    
    
            
            # Get the dual variables
            pi = []    
            for name, constraint in prob.constraints.items():
                pi.append(constraint.pi)
            
            pi = np.array(pi)
            
            if self.debug:
                print("objective=", lp.value(prob.objective))
                print ("dual objective = ", (h - T @ self.x_nu[-1]) @ pi)
                print ("dual variables = ", pi)

            E += self.p[k] * pi.T @ T
            e += self.p[k] * pi.T @ h
              
        w_nu = e - E @ self.x_nu[-1]
        
        if (np.abs(self.theta_nu[-1] - w_nu) <= self.precision 
            and self.theta_nu[-1] is not None):
            # The solution is optimal
            return 0 # no cut needed, solution is optimal
        
        # Else append optimality cut
        if self.verbose:
            print ("w_nu = ", w_nu)
            print ("theta_nu = ", self.theta_nu[-1])
        print ("Optimality cut made")
        print ("E = ", E)
        print ("e = ", e)
        self.s += 1
        self.E.append(E)
        self.e.append(e)
        return 1 # a cut was made
    

        
