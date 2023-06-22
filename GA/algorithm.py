import numpy as np
from typing import List, Tuple
from copy import deepcopy
import multiprocessing as mp
from itertools import starmap
from .selection import roulette_wheel_selection
import multiprocessing as mp
import itertools


#this wrapper is needed in order to enable paralelization
def f_wrapper(i : int, ff, individual, *args): return i, ff(individual,*args)


def ga(
        initial_population : List,
        fitness_callback_tuple : Tuple,
        selection_callback_tuple : Tuple,
        crossover_callback_tuple : Tuple,
        mutation_callback_tuple_list : List[Tuple],
        elite_fraction : float,
        number_of_generations : int
        ):
    
    population = [ deepcopy(individual) for individual in initial_population ]
    population_size = len(population)

    assert elite_fraction >= 0 and elite_fraction < 1.

    number_of_elite = max(1, int(population_size * elite_fraction))
    number_of_offspring = population_size - number_of_elite

    fitness_fn, *opt_fitness_fn_args = fitness_callback_tuple
    selection_fn, *opt_selection_fn_args = selection_callback_tuple
    crossover_fn, *opt_crossover_fn_args = crossover_callback_tuple

    history = []
    stats_history = []


    try:
        for g in range(number_of_generations):

            pool = mp.Pool(mp.cpu_count())
            tmp = []

            #calculate fitness for each individual 
            for i, individual in enumerate(population):
                #fitness = fitness_fn(individual, *opt_fitness_fn_args)
                #population[i] = individual, fitness
                pool.apply_async(f_wrapper, args=(i, fitness_fn, individual, *opt_fitness_fn_args), callback= lambda res: tmp.append(res) )

            pool.close()
            pool.join()

            for i, f in tmp:
                individual = population[i]
                population[i] = individual, f
    
            #sort population acording to fitness
            population.sort(key = lambda ind: ind[1], reverse= True)

            #split individuals and their fitness
            individuals, fitnesses = zip(*population)
            normalized_fitnesses = [f / fitnesses[0] for f in fitnesses]

            best_individual = individuals[0]
            history += [best_individual]

            best_fitness = fitnesses[0]
            Q1 = np.quantile(normalized_fitnesses, 0.75)
            median = np.median(normalized_fitnesses)
            Q3 = np.quantile(normalized_fitnesses, 0.25)
            worst_fitness = normalized_fitnesses[-1]
            avg_fitness = np.average(normalized_fitnesses)
            fitness_stddev = np.std(normalized_fitnesses)

            stats_history += [(best_fitness, worst_fitness, avg_fitness, fitness_stddev, Q1, median, Q3)]

            sinfo = f'''
            generation {g}:
                best fitness:   {best_fitness}
                Q1:             {Q1}
                median fitness: {median}
                Q3:             {Q3}
                worst fitness:  {worst_fitness}
                avg fitness:    {avg_fitness}
                fitness stddev: {fitness_stddev}
            '''
            print(sinfo)
        
            #create elite that will be preserved in the next generation
            elite = individuals[:number_of_elite]

            #get list of tuples of parents indices from selection function
            parents = selection_fn(fitnesses, number_of_offspring, *opt_selection_fn_args)
        
            #create offsprings
            offsprings = list(starmap(crossover_fn, ((individuals[p1_i], individuals[p2_i], *opt_crossover_fn_args) for p1_i, p2_i in parents)))

            #
            for i in range(number_of_offspring):
                for op, *op_args in mutation_callback_tuple_list:
                    offsprings[i] = op(offsprings[i], *op_args)

            #merge elite and offsprings to create next generation
            population = [ individual for individual in itertools.chain(elite, offsprings) ]
    except KeyboardInterrupt:
        return history, stats_history
    except TypeError:
        return history, stats_history

    return history, stats_history


if __name__ == "__main__":
    pop = [ np.random.rand() for _ in range(4) ]

    ga(
        pop,
        (lambda x: x,),
        (roulette_wheel_selection,),
        (lambda x, y: x,),
        [],
        0.1, 1 )

    pass