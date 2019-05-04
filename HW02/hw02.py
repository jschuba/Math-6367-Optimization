import numpy as np
import pandas as pd


def backward_dp(failure_prob):
    f = []
    f_arg = []
    for k in range(len(failure_prob)-1, -1, -1):
        fk = []
        fk_arg = []
        for i in range(len(failure_prob[0])):
            if k == len(failure_prob)-1:
                fk.append(failure_prob[k][i])
                fk_arg.append(i)
                print(f"f_{k}[{i}] = {failure_prob[k][i]}")
            else:
                if i == 0:
                    t = 0
                    temp = failure_prob[k][t]*f[-1][i-t]
                    fk.append(temp)
                    fk_arg.append(0)
                else:
                    temp = []
                    for t in range(i):
                        temp.append(failure_prob[k][t]*f[-1][i-t])
                    fk.append(min(temp))
                    fk_arg.append(np.argmin(temp))
                print(f"f_{k}[{i}] = min({np.round(temp, 4)}) = {np.round(fk[-1],4)}  ")
                print(f"\t argmin = {fk_arg[-1]}")
                
        f.append(fk)
        f_arg.append(fk_arg)
    return f, f_arg

def forward_strategy(f, f_arg):
    prob_of_failing_all_courses = f[-1][-1]
    
    strategy = [0 for _ in f]
    
    hours_remaining = len(f[0])-1
    print(f"Working on forward strategy")
    
    for k in range(len(f)-1, -1, -1):
        print(f"{hours_remaining} hours remaining")
        print(f"Considering course {k}")
        hours_to_spend = f_arg[k][hours_remaining]
        print(f"Spending {hours_to_spend} hours")
        strategy[k] = hours_to_spend
        hours_remaining -= hours_to_spend
        
    
    strategy.reverse()
    return prob_of_failing_all_courses, strategy
    
def print_failure_matrix(failure_matrix, course_list):
    df = pd.DataFrame(failure_matrix)
    df.columns = [i for i in range(len(failure_matrix[-1]))]
    df.index = course_list
    print("The course-failure probability matrix is:")
    print(df)
    print()
    
#if __name__ == "__main__":
#    main()
    
def check(failure_matrix, strategy):
    prob_of_failing_all_courses = 1
    for course, hours  in enumerate(strategy):
        prob_of_failing_all_courses *= failure_matrix[course][hours]
    return prob_of_failing_all_courses

courses = ["Algebra", "Geometry", "Optimization"]
print("Note: The courses are indexed:")
for i, course in enumerate(courses):
    print(f"{i}: {course}")
print() 

failure_prob = [[0.8, 0.75, 0.90],
                [0.70, 0.70, 0.70],
                [0.65, 0.67, 0.60],
                [0.62, 0.65, 0.55],
                [0.60, 0.62, 0.50]]
# Transpose the failure matrix, so that we can index it by [course][hours]
failure_prob = np.transpose(failure_prob)

print_failure_matrix(failure_prob, courses)

f, f_arg = backward_dp(failure_prob)

prob_of_failing_all_courses, strategy = forward_strategy(f, f_arg)

check_prob = check(failure_prob, strategy)

print(f"The probability of failing all courses is: {prob_of_failing_all_courses}")
print("The optimal strategy is to spend:")
for course, hours in zip(courses, strategy):
    print(f"{hours} hours(s) on {course}")