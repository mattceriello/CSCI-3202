from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
import random
from random import choices 
import math 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams

from matplotlib import pyplot as plt 

class GASolver: 
    def __init__(self, params, lst_of_identifiers, n):
        # Parameters for GA: see geneticAlgParams
        # Also includes test data for regression and checking validity
        self.params = params
        # The population size 
        self.N = n
        # Store the actual population (you can use other data structures if you wish)
        self.pop = []
        # A list of identifiers for the expressions
        self.identifiers = lst_of_identifiers
        # Maintain statistics on best fitness in each generation
        self.population_stats = []
        # Store best solution so far across all generations
        self.best_solution_so_far = None
        # Store the best fitness so far across all generations
        self.best_fitness_so_far = -float('inf')

    # Please add whatever helper functions you wish.
    
    
    # TODO: Implement the genetic algorithm as described in the
    # project instructions.
    # This function need not return anything. However, it should
    # update the fields best_solution_so_far, best_fitness_so_far and
    # population_stats
    def run_ga_iterations(self, n_iter=1000):
        x = 0
        while(x < self.N):
            exp = generate_random_expr(self.params.depth, self.identifiers, self.params)
            if is_viable_expr(exp, self.identifiers, self.params) == True:
                self.pop.append(exp)
                x+=1
                
        cum_weights = []     
        for w in range(0,self.N):
            fit = compute_fitness(self.pop[w],self.identifiers, self.params)
            prob = math.exp(fit/self.params.temperature)
            cum_weights.append(prob)
        
        curpop = self.pop
        for i in range(0,n_iter):
            k = self.params.elitism_fraction *self.N
            self.pop.sort(key = lambda x: compute_fitness(x, self.identifiers, self.params), reverse = True)         
            next_gen = []
        
            for k in range(0,int(k)):
                next_gen.append(self.pop[k])
            
            count = k+1
            while count < self.N:            
                e1 = 0
                e2 = 0
                f1 = 0
                f2 = 0
                lst = random.choices(self.pop,cum_weights, k = 2)
                e1 = lst[0]
                e2 = lst[1]
                e1_cross,e2_cross = random_subtree_crossover(e1,e2,copy = True) 
                e1_mut = random_expression_mutation(e1_cross,self.identifiers, self.params)
                f1 = compute_fitness(e1_mut,self.identifiers, self.params)
                e2_mut = random_expression_mutation(e2_cross,self.identifiers, self.params)
                f2 = compute_fitness(e2_mut,self.identifiers, self.params)                
                if is_viable_expr(e1_mut, self.identifiers, self.params) == True and count < self.N:
                    next_gen.append(e1_mut)
                    count+=1
                if is_viable_expr(e2_mut, self.identifiers, self.params) == True and count < self.N: 
                    next_gen.append(e2_mut)
                    count+=1
        
            next_gen_fit = []    
            for w in range(len(next_gen)):
                val = (next_gen[w], compute_fitness(next_gen[w],self.identifiers, self.params))
                next_gen_fit.append(val)
            
            best_fit = -10000000
            best_fit_idx = 0
            for v in range(len(next_gen_fit)):
                fitness = list(next_gen_fit[v])
                solution = fitness[0]
                fitness_val = fitness[1]
                if fitness_val > best_fit:
                    best_fit = fitness_val
                    best_fit_idx = v
  
            #append best fitness
            self.best_fitness_so_far = best_fit
        
            #append best solution
            self.best_solution_so_far = next_gen[best_fit_idx]
            self.population_stats.append(best_fit)        
            self.pop = next_gen
         
        #raise NotImplementedError('Not implemented yet!')

## Function: curve_fit_using_genetic_algorithms
# Run curvefitting using given parameters and return best result, best fitness and population statistics.
# DO NOT MODIFY
def curve_fit_using_genetic_algorithm(params, lst_of_identifiers, pop_size, num_iters):
    solver = GASolver(params, lst_of_identifiers, pop_size)
    solver.run_ga_iterations(num_iters)
    return (solver.best_solution_so_far, solver.best_fitness_so_far, solver.population_stats)


# Run test on a toy problem.
if __name__ == '__main__':
    params = GAParams()
    params.regression_training_data = [
       ([-2.0 + 0.02*j], 5.0 * math.cos(-2.0 + 0.02*j) - math.sin((-2.0 + 0.02*j)/10.0)) for j in range(201)
    ]
    params.test_points = list([ [-4.0 + 0.02 * j] for j in range(401)])
    solver = GASolver(params,['x'],500)
    solver.run_ga_iterations(100)
    print('Done!')
    print(f'Best solution found: {solver.best_solution_so_far.simplify()}, fitness = {solver.best_fitness_so_far}')
    stats = solver.population_stats
    niters = len(stats)
    plt.plot(range(niters), [st[0] for st in stats] , 'b-')
    plt.show()



