import GA.crossover
from .neural_network import NeuralNetwork
import numpy as np


def unifrom_weights_crossover(parent1 : NeuralNetwork, parent2 : NeuralNetwork) -> NeuralNetwork:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()

    for i, _ in enumerate(offspring.weights):
        offspring.weights[i] = GA.crossover.uniform_crossover(parent1.weights[i], parent2.weights[i])

    for i, _ in enumerate(offspring.biases):
        offspring.biases[i] = GA.crossover.uniform_crossover(parent1.biases[i], parent2.biases[i])
    
    return offspring


def uniform_neurons_crossover(parent1 : NeuralNetwork, parent2 : NeuralNetwork) -> NeuralNetwork:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()

    for i, W in enumerate(offspring.weights):
        R, C = W.shape
        for r in range(R):
            if np.random.rand() < 0.5:
                offspring.biases[i][r] = parent2.biases[i][r]
                offspring.weights[i][r] = parent2.weights[i][r]
                
    return offspring


def simulated_crossover(parent1 : NeuralNetwork, parent2 : NeuralNetwork, eta : float) -> NeuralNetwork:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()

    for i, _ in enumerate(offspring.weights):
        offspring.weights[i] = GA.crossover.simulated_crossover(parent1.weights[i], parent2.weights[i], eta)

    for i, _ in enumerate(offspring.biases):
        offspring.biases[i] = GA.crossover.simulated_crossover(parent1.biases[i], parent2.biases[i], eta)
        
    return offspring


def single_point_crossover(parent1 : NeuralNetwork, parent2 : NeuralNetwork) -> NeuralNetwork:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()

    for i, _ in enumerate(offspring.weights):
        offspring.weights[i] = GA.crossover.single_point_crossover(parent1.weights[i], parent2.weights[i])

    for i, _ in enumerate(offspring.biases):
        offspring.biases[i] = GA.crossover.single_point_crossover(parent1.biases[i], parent2.biases[i])
        
    return offspring


def double_point_crossover(parent1 : NeuralNetwork, parent2 : NeuralNetwork) -> NeuralNetwork:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()

    for i, _ in enumerate(offspring.weights):
        offspring.weights[i] = GA.crossover.double_point_crossover(parent1.weights[i], parent2.weights[i])

    for i, _ in enumerate(offspring.biases):
        offspring.biases[i] = GA.crossover.double_point_crossover(parent1.biases[i], parent2.biases[i])
        
    return offspring