import numpy as np
from typing import List, Tuple


#https://github.com/Chrispresso/SnakeAI/blob/master/genetic_algorithm/crossover.py
def simulated_crossover(parent1 : np.ndarray, parent2 : np.ndarray, eta : float) -> np.ndarray:

    # Calculate Gamma (Eq. 9.11)
    rand = np.random.random(parent1.shape)
    gamma = np.empty(parent1.shape)
    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # First case of equation 9.11
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Second case

    # Calculate Child 1 chromosome (Eq. 9.9)
    chromosome1 = 0.5 * ((1 + gamma)*parent1 + (1 - gamma)*parent2)
    # Calculate Child 2 chromosome (Eq. 9.10)
    #chromosome2 = 0.5 * ((1 - gamma)*parent1 + (1 + gamma)*parent2)
    return chromosome1


def uniform_crossover(parent1 : np.ndarray, parent2 : np.ndarray) -> np.ndarray:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()
    
    mask = np.random.uniform(0, 1, size= parent1.shape)

    offspring[mask > 0.5] = parent2[mask > 0.5]

    return offspring


def single_point_crossover(parent1 : np.ndarray, parent2 : np.ndarray) -> np.ndarray:
    assert parent1.shape == parent2.shape

    flat_p1 = parent1.flatten()
    flat_p2 = parent2.flatten()

    offspring = flat_p1.copy()

    break_point = np.random.randint(len(offspring))

    offspring[break_point:] = flat_p2[break_point:]

    return offspring.reshape(parent1.shape)


def double_point_crossover(parent1 : np.ndarray, parent2 : np.ndarray) -> np.ndarray:
    assert parent1.shape == parent2.shape
    
    flat_p1 = parent1.flatten()
    flat_p2 = parent2.flatten()

    offspring = flat_p1.copy()

    break_point_1 = np.random.randint(len(offspring))
    break_point_2 = np.random.randint(break_point_1, len(offspring))

    offspring[break_point_1:break_point_2] = flat_p2[break_point_1:break_point_2]
    
    return offspring.reshape(parent1.shape)