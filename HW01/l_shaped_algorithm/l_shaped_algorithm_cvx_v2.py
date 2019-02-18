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


class L_Shaped_Algorithm():
    """
    
    """
    
    def __init__(self, c, A_eq, b_eq, A_ineq, b_ineq, W, h_driver, T_driver, q, 
                 realizations, probabilities, 
                 max_iter = 100, precision=10e-6, 
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
        self.D_list = []         # list of matrices for feasibility cuts
        self.d_list = []         # list of vectors for feasibility cuts
        
        ''' 
        Matrices and vectors which will form constraints pertaining to 
        optimality cuts, ie:
            E[i] @ x >= e[i]  where 1 <= i <= s
        '''
        self.E_list = []         # list of matrices for optimality cuts
        self.e_list = []         # list of vectors for optimality cuts
        
        ''' 
        Lists to hold the values obtained in each iteration 
        '''
        self.x_nu_list = []
        self.theta_nu_list = []
        self.objective_value_list = []
        
        
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
                self.value = np.round(self.objective_value_list[-1], round_precision)
                self.solution = np.round(self.x_nu_list[-1], round_precision)
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
        
        x = cp.Variable(n)
        theta = cp.Variable(1)
        
        if self.s == 0:
            # There are no optimality cuts, so set theta to -inf
            try: 
                objective = cp.Minimize(self.c @ x)
            except ValueError:
                objective = cp.Minimize(self.c * x)
            theta_solution = -np.inf
        else:
            try: 
                objective = cp.Minimize(self.c @ x + theta)
            except ValueError:
                objective = cp.Minimize(self.c * x + theta)
        
        constraints = [x >= 0]
        
        if self.A_eq is not None:
            # We must append the equality constraints on x
            try:
                constraints.append( self.A_eq @ x == self.b_eq )
            except ValueError:
                constraints.append( self.A_eq * x == self.b_eq )
        if self.A_ineq is not None:
            # We must append the inequality constraints on x
            try:
                constraints.append( self.A_ineq @ x <= self.b_ineq )
            except ValueError:
                constraints.append( self.A_ineq * x <= self.b_ineq )
                       
        for r in range(len(self.D_list)):
            # add constraints for each feasibility cut
            try:
                constraints.append (self.D_list[r] @ x >= self.d_list[r] )
            except ValueError:
                constraints.append (self.D_list[r] * x >= self.d_list[r] )
        for s in range(len(self.E_list)):
            # add constraints for each optimality cut
            try:
                constraints.append( self.E_list[s] @ x + theta >= self.e_list[s] )
            except ValueError:
                constraints.append( self.E_list[s] * x + theta >= self.e_list[s] )
                
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=self.verbose)
            
        if result is None and self.nu == 1:
            self.objective_value_list.append(0)
            self.x_nu_list.append(np.zeros(self.c.shape))
            self.theta_nu_list.append(-np.inf)
            return 1
        
        # CVX sometimes makes the variables into funny size matrices, so we 
        # need to make them n-by-1 vectors
        x_solution = np.array([x.value])
        x_solution = x_solution.reshape(x_solution.size)
        
        if self.s > 0:   
            theta_solution = theta.value
        
        print ("objective value = ", result)
        print ("x_nu            = ", x_solution)
        print ("theta_nu        = ", theta_solution)
            
        self.objective_value_list.append(result)
        self.x_nu_list.append(x_solution)
        self.theta_nu_list.append(theta_solution)
        
        return 1
        

    def step_2(self):
        '''
        Solve LPs for each possible realization of the random variables, and 
        make feasibility cuts as appropriate
        '''        
        
        print ()
        print (f"----------------- Step 2 -----------------")
        
        n = self.W.shape[0]
        if len(self.W.shape) > 1:
            m = self.W.shape[1]
        else:
            m = 1
        
        for k in range(self.K):
            vp = cp.Variable(n)
            vm = cp.Variable(n)
            y = cp.Variable(m)

            objective = cp.Minimize(cp.sum_entries(vp) + cp.sum_entries(vm))            
            
            # We use the user-specified driver functions to get the correct
            # matrix T and h for this particular realization of the random
            # variables
            T = self.T_driver(self.x_nu_list[-1], self.realizations[k])
            h = self.h_driver(self.x_nu_list[-1], self.realizations[k])
            
            try:
                constraints = [self.W @ y + vp - vm == h - T @ self.x_nu_list[-1],
                               vp >= 0,
                               vm >= 0,
                               y >= 0]
            except ValueError:
                constraints = [self.W * y + vp - vm == h - T * self.x_nu_list[-1],
                               vp >= 0,
                               vm >= 0,
                               y >= 0]
                           
            prob = cp.Problem(objective, constraints)
            result = prob.solve(verbose=self.verbose)
           
            if np.abs(result) > self.precision:
                # Then we need to add a feasibility cut
                self.r += 1
                
                # Get the dual variables
                sigma = -1 * constraints[0].dual_value
                sigma = np.array(sigma).reshape(sigma.size)
                
                print ("Feasibility cut identified")
                print ("objective      = ", result)
                print ("dual objective = ", (h - T @ self.x_nu_list[-1]) @ sigma)
                print ("dual variables = ", sigma)
                
                D = sigma.T @ T
                d = sigma.T @ h
                print ("Dk = ", D)
                print ("dk = ", d)
                self.D_list.append(D)
                self.d_list.append(d)
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
        E = np.zeros(len(self.x_nu_list[-1]))
        e = 0
        
        for k in range(self.K):
            y = cp.Variable(m)

            # We use the user-specified driver functions to get the correct
            # matrix T and h for this particular realization of the random
            # variables
            T = self.T_driver(self.x_nu_list[-1], self.realizations[k])
            h = self.h_driver(self.x_nu_list[-1], self.realizations[k])

            # Define the objective function and constraints
            try:
                objective = cp.Minimize(self.q[k] @ y[0:len(self.q[k])])
                constraints = [self.W @ y == h - T @ self.x_nu_list[-1],
                               y >= 0]
            except ValueError:
                objective = cp.Minimize(self.q[k] * y[0:len(self.q[k])])
                constraints = [self.W * y == h - T @ self.x_nu_list[-1],
                               y >= 0]
            
            prob = cp.Problem(objective, constraints)
            result = prob.solve(verbose=self.verbose)
                       
            # Get the dual variables
            pi = -1 * np.array(constraints[0].dual_value)
            pi = np.array(pi).reshape(pi.size)
                        
            if self.debug:
                print("objective       = ", result)
                print ("dual objective = ", (h - T @ self.x_nu_list[-1]) @ pi)
                print ("dual variables = ", pi)

            E += self.p[k] * pi.T @ T
            e += self.p[k] * pi.T @ h
        
        w_nu = e - E @ self.x_nu_list[-1]
        
        if (np.abs(self.theta_nu_list[-1] - w_nu) <= self.precision):
            # The solution is optimal
            return 0 # no cut needed, solution is optimal
        
        # Else append optimality cut
        if self.verbose:
            print ("w_nu = ", w_nu)
            print ("theta_nu = ", self.theta_nu_list[-1])
        print ("Optimality cut made")
        print ("E = ", E)
        print ("e = ", e)
        self.s += 1
        self.E_list.append(E)
        self.e_list.append(e)
        return 1 # a cut was made
    

        
