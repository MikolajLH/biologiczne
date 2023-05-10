import numpy as np
from copy import deepcopy
from collections.abc import Callable
import multiprocessing as mp

#dummy classes for type hints
class Entity: pass
class Args: pass

# initial_population - list of entities that will be deepcopied to create population used in genetic algorithm
#
# fitness_function - function that, the algorithm will maximize, it takes entity end returns it's fitness
#
# selection_function_info - e.g (rank_selection, 1, 1.5) or (roulette_wheel_selection,)
#   this argument is passed in a form of a tuple,
#   first element of a tuple is a selection function to be used - (rank_selection, roulette_wheel_selection, tournament_selection)
#       this function is expected to take list of pairs in a form (entity, fitness) as it's first argument and return list of indices interpreted as mating pool
#   remaining elements of this tuple are interpreted as arguments to be passed to selection function after the first argument
#
# crossover_function_info - function responsible for crossover operation, it is passed in an analogous way as selection function 
#
# mutation_funcions_info - list of mutation operators, each element of this list is passed in an analogous way as in selection_function_info and crossover_function_info
#
# elite_percent - percent of best individuals in each generatoin that will be preserved in next generation,
# the best individual is always preserved regardless of this parameter
#
# number_of_generations - self explanatory
#

#this wrapper is needed in order to enable paralelization
def f_wrapper(i : int, ff, e, *args):
    return i, *(ff(e,*args))


def genetic_algorithm(
        initial_population : list[Entity],
        fitness_function_info : tuple[Callable[[Entity, Args], float], Args],
        selection_function_info : tuple[ Callable[ [list[tuple[Entity, float], Args ] ], list[int] ], Args ],
        crossover_function_info : tuple[ Callable[[Entity, Entity, Args], Entity], Args ],
        mutation_functions_info : list[ tuple[Callable[ [Entity, Args], Entity], Args] ],
        elite_percent : float,
        number_of_generations : int,
        fitness_translation_function = lambda x: x,
        extra_info_function = lambda *x: 'N/A'
        ) -> tuple[list[tuple[Entity, float]], list[tuple[float,float,float,float,float]]]:

    elite_percent = max(0, min(1, elite_percent))

    population = [(deepcopy(e), 0) for e in initial_population]

    #populations size
    N = len(population)

    fitness_function, *fitness_function_args = fitness_function_info
    selection_function, *selection_function_args = selection_function_info
    crossover_function, *crossover_function_args = crossover_function_info


    #winners write history
    history = []
    history_statistics = []

    try:
        for g in range(number_of_generations):

            pool = mp.Pool(mp.cpu_count())
            tmp = []
            #calculate fitness for each individual
            for i, (e, f) in enumerate(population):
                #f = fitness_function(e, *fitness_function_args)
                #population[i] = (e,f)
                pool.apply_async(f_wrapper, args=(i, fitness_function, e, *fitness_function_args), callback= lambda res: tmp.append(res) )

            pool.close()
            pool.join()

            for i, f, *extra in tmp:
                e, _ = population[i]
                population[i] = (e, f, *extra)


            #sort the population, with decreasing fitness
            population.sort(key= lambda pair: pair[1], reverse= True)

            best_e, best_f, *extra_info = population[0]

            # get rid of potential extra information returned by fitness function
            for i, (e, f,*_) in enumerate(population):
                population[i] = e, f


            #statistical information about current generation
            fs = [fitness_translation_function(f)  for _,f in population]

            Q1_f = np.quantile(fs, 0.25)
            Q3_f = np.quantile(fs, 0.75)
            best_f = fitness_translation_function(best_f)
            total_f = sum(fitness_translation_function(f) for _, f in population)
            avg_f = total_f / N
            median_f = fitness_translation_function(population[N//2][1])
            worst_f = fitness_translation_function(population[-1][1])
            f_std = np.std(fs)


            #add best individual in this generation to the history list
            history += [(best_e, best_f)]

            #history_statistics += [SimpleNamespace(generation= g, best_fitness= best_f, worst_fitness= worst_f, avg_fitness= avg_f, median_fitness= median_f, fitness_std= f_std)]
            history_statistics += [(g, best_f, worst_f, avg_f, Q1_f, median_f, Q3_f, f_std)]

            sinfo = f'''
            generation {g}:
                the best info:  {extra_info_function(*extra_info)}
                best fitness:   {best_f}
                Q1:             {Q1_f}
                median fitness: {median_f}
                Q3:             {Q3_f}
                worst fitness:  {worst_f}
                avg fitness:    {avg_f}
                fitness stddev: {f_std}
            '''
            print(sinfo)

            #creating a mating pool
            mating_pool_indices = selection_function(population, *selection_function_args)
        

            #crossover operation
            elite_number = max(1, int(elite_percent * N))
            children_number = N - elite_number

            children = []

            for _ in range(children_number):
                m_i, f_i = np.random.choice(mating_pool_indices, 2)
                m, _ = population[m_i]
                f, _ = population[f_i]
                children += [(crossover_function(m, f, *crossover_function_args), 0)]

            #mutation operations
            for i, _ in enumerate(children):
                for op, *op_args in mutation_functions_info:
                    e, _ = children[i]
                    children[i] = op(e, *op_args), f

            #creation of population for next generation
            population = population[:elite_number] + children
    except KeyboardInterrupt:
        return history, history_statistics
    
    return history, history_statistics


#types of selection briefly explained
#https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)

def roulette_wheel_selection(population : list[tuple[Entity, float]], mating_pool_percent_size : float = 1) -> list:

    N = len(population)
    total = sum(f for _,f in population)
    probabilities = [f / total for _, f in population]
    M = int(N * mating_pool_percent_size)
    print(f"max probability on roulette wheel: {probabilities[0]}")

    return np.random.choice(N, M, p=probabilities)


def rank_selection(population : list, mating_pool_percent_size : float = 1, sp = None) -> list:
    N = len(population)

    if sp is None:
        total = (1 + N) * N / 2
        probabilities = [r / total for r in range(N, 0, -1)]
    else:
        if not (sp >= 1. and sp <= 2.):
            raise RuntimeError("sp has to be in range [1.0, 2.0] ")
        probabilities = [(1 / N) * (sp - (2*sp - 2)*(i - 1)/(N - 1)) for i in range(1, N + 1)][::-1]

    M = int(N * mating_pool_percent_size)
    return np.random.choice(N, M, p= probabilities)


def tournament_selection(population : list, mating_pool_percent_size : float, tournament_size : int):
    N = len(population)
    M = int(N * mating_pool_percent_size)

    mating_pool = []
    for _ in range(M):
        participants_indices = np.random.choice(N, tournament_size, replace=False)
        best_index = 0
        _, best_fitness = population[best_index]
        for i in participants_indices:
            _, fitness = population[i]
            if fitness > best_fitness:
                best_index, best_fitness = i, fitness
        mating_pool += [best_index]

    return mating_pool



# [A, A, A, A, A] & [B, B, B, B, B] -> [A, A, A, B, B]
def single_point_crossover(m : np.array, f : np.array):
    assert len(m) == len(f)
    break_point = np.random.randint(len(m))

    return np.concatenate([m[:break_point], f[break_point:]])

# [A, A, A, A, A] & [B, B, B, B, B] -> [A, B, B, A, A]
def double_point_crossover(m, f):
    assert len(m) == len(f)
    bp_1 = np.random.randint(len(m))
    bp_2 = np.random.randint(len(m))
    if bp_1 > bp_2:
        bp_1, bp_2 = bp_2, bp_1

    return np.concatenate([m[:bp_1], f[bp_1:bp_2], m[bp_2:]])



def uniform_additive_mutation(v, rate : float, a, b):
    q = deepcopy(v)
    for i in range(len(v)):
        if np.random.rand() < rate:
            q[i] += np.random.uniform(a, b)
    return q

def gaussian_additive_mutation(v, rate : float, mean : float = 0, stddev : float = 1):
    q = deepcopy(v)
    for i in range(len(v)):
        if np.random.rand() < rate:
            q[i] += np.random.normal(mean, stddev)
    return q


def uniform_multiplicative_mutation(v, rate : float, a, b):
    q = deepcopy(v)
    for i in range(len(v)):
        if np.random.rand() < rate:
            q[i] *= np.random.uniform(a, b)
    return q


def gaussian_multiplicative_mutation(v, rate : float, mean : float = 1, stddev : float = 1):
    q = deepcopy(v)
    for i in range(len(v)):
        if np.random.rand() < rate:
            q[i] *= np.random.normal(mean, stddev)
    return q
