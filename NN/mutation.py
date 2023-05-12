import GA.mutation
from .neural_network import NeuralNetwork
import numpy as np


def gaussian_mutation(chromosome : NeuralNetwork, mutation_prob : float, mean : float = 0, stddev : float = 1, scale : float = 1) -> NeuralNetwork:
    assert mutation_prob > 0 and mutation_prob < 1
    new_chromosome = chromosome.copy()
    for i, W in enumerate(new_chromosome.weights):
        new_chromosome.weights[i] = GA.mutation.gaussian_mutation(W, mutation_prob, mean, stddev, scale)

    for i, B in enumerate(new_chromosome.biases):
        new_chromosome.biases[i] = GA.mutation.gaussian_mutation(B, mutation_prob, mean ,stddev, scale)

    return new_chromosome


def uniform_mutation(chromosome : NeuralNetwork, mutation_prob : float, low : float, high : float) -> NeuralNetwork:
    assert mutation_prob > 0 and mutation_prob < 1
    new_chromosome = chromosome.copy()
    for i, W in enumerate(new_chromosome.weights):
        new_chromosome.weights[i] = GA.mutation.uniform_mutation(W, mutation_prob, low, high)
    
    for i, B in enumerate(new_chromosome.biases):
        new_chromosome.biases[i] = GA.mutation.uniform_mutation(B, mutation_prob, low, high)

    return new_chromosome


def gaussian_neuron_mutation(chromosome : NeuralNetwork, mutation_prob : float, k : int, mean : float = 0, stddev : float = 1, scale : float = 1) -> NeuralNetwork:
    new_chromosome = chromosome.copy()
    pass