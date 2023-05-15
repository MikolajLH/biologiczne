import numpy as np
import pygame
import os
from snake.raw_snake import RawSnake
from snake.utils.direction import Direction
from snake.utils.game_status import GameStatus
import matplotlib.pyplot as plt
from NN.neural_network import NeuralNetwork
from NN.activation_functions import *
import GA.algorithm
import GA.selection
import NN.mutation
import NN.crossover



class HiddenPlayer:
    def __init__(self, brain : NeuralNetwork) -> None:

        self.__brain = brain
        self.__snake = RawSnake(False)

    def set_brain(self, brain : NeuralNetwork) -> None:
        self.__brain = brain

    
    def play(self):

        self.__snake.new_game()
        steps = 0

        while self.__snake.getGameStatus() == GameStatus.GAME_ON:
            sensors = self.__snake.getSensors()

            inputVector = np.array(sensors.toNormalizedList())

            outputVector = self.__brain(inputVector)

            next_move = Direction(np.argmax(outputVector))

            self.__snake.make_move(next_move)

            steps += 1.

        apples = self.__snake.getScore()

        #return 2**apples - 1, apples, steps
        return steps + (2**apples + 500 * apples**2.1) - ((apples**1.2) * (0.25 * steps)**1.3), apples, steps



def snake_fitness(nn : NeuralNetwork, player : HiddenPlayer, lifes : float = 1):
    assert lifes > 0

    player.set_brain(nn)
    fitness, apples, steps = sum(np.array(player.play()) for _ in range(lifes) ) / lifes

    nn.apples = apples
    nn.steps = steps

    return fitness


if __name__ == "__main__":
    pygame.init()
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

    model = NeuralNetwork()
    model.add_input_layer(32)
    model.add_hidden_layer(24, relu)
    model.add_hidden_layer(12, relu)
    model.add_output_layer(4, softmax)

    player = HiddenPlayer(model)

    pop_size = 500
    init_pop = [model.g_rand() for _ in range(pop_size)]

    evolution_history, statistics = GA.algorithm.ga(
        init_pop,
        (snake_fitness, player),
        (GA.selection.roulette_wheel_selection,),
        (NN.crossover.unifrom_weights_crossover,),
        [(NN.mutation.gaussian_mutation,0.05)],
        0.05, 50)
    
    best_f, worst_f, avg_f, std_f, Q1_f, median_f, Q3_f = zip(*statistics)

    best_snake = evolution_history[-1]
    best_snake.save("sensor_test")

    # plt.plot(best_f, label="best fitness")
    # plt.semilogy()
    # plt.legend()
    # plt.show()
    #
    # plt.plot(worst_f, label= "worst fitness")
    # plt.plot(avg_f, label= "avg fitness")
    # plt.plot(Q1_f, label= "Q1")
    # plt.plot(Q3_f, label= "Q3")
    # plt.plot(median_f, label= "median")
    # plt.plot(std_f, label= "stddev")
    # plt.legend()
    # plt.show()


