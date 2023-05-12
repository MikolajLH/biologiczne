from typing import List, Tuple
from numpy.random import choice


def roulette_wheel_selection(population_fitnesses : List[float], number_of_offspring : int) -> List[Tuple[int,int]]:
    assert number_of_offspring > 0 and number_of_offspring < len(population_fitnesses)
    total_fitness = sum(fitness for fitness in population_fitnesses)
    probabilities = [fitness / total_fitness for fitness in population_fitnesses]

    return [tuple(choice(range(len(population_fitnesses)), 2, p=probabilities)) for _ in range(number_of_offspring)]


def tournament_selection(population_fitnesses : List[float], number_of_offspring : int, tournament_size : int = 2) -> List[Tuple[int,int]]:
    assert number_of_offspring > 0 and number_of_offspring < len(population_fitnesses)
    assert tournament_size > 0 and tournament_size < len(population_fitnesses)
    parents = []
    for _ in range(number_of_offspring):
        a, b = choice(len(population_fitnesses), tournament_size)
        p1 = a if population_fitnesses[a] > population_fitnesses[b] else b

        a, b = choice(len(population_fitnesses), tournament_size)
        p2 = a if population_fitnesses[a] > population_fitnesses[b] else b

        parents += [(p1,p2)]

    return parents


def rank_selection(population_fitnesses : List[float], number_of_offspring : int) -> List[Tuple[int,int]]:
    N = len(population_fitnesses)
    total = (1 + N) * N / 2
    probabilities = [r / total for r in range(N, 0, -1)]

    return [tuple(choice(range(len(population_fitnesses)), 2, p=probabilities)) for _ in range(number_of_offspring)]
