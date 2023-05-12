import numpy as np


def gaussian_mutation(chromosome : np.ndarray, mutation_prob : float, mean : float = 0, stddev : float = 1, scale : float = 1) -> np.ndarray:
    assert mutation_prob > 0 and mutation_prob < 1

    mutation_array = np.random.random(chromosome.shape) < mutation_prob
    new_chromosome = chromosome.copy()
    new_chromosome[mutation_array] += scale * np.random.normal(mean, stddev, chromosome.shape)[mutation_array]
    return new_chromosome

def uniform_mutation(chromosome : np.ndarray, mutation_prob : float, low : float, high : float) -> np.ndarray:
    assert mutation_prob > 0 and mutation_prob < 1

    mutation_array = np.random.random(chromosome.shape) < mutation_prob
    new_chromosome = chromosome.copy()
    new_chromosome[mutation_array] += np.random.uniform(low, high, chromosome.shape)[mutation_array]
    
    return new_chromosome


if __name__ == "__main__":

    a = np.zeros(10)

    print(a)
    print(gaussian_mutation(a, 0.5))
    print(a)
    
    pass