import NN
import GA
import numpy as np
from random import randint
import time
import pygame
import matplotlib.pyplot as plt

from snake.snake_handler import SnakeHandler
from snake.raw_snake import RawSnake
from snake.utils.direction import Direction
from snake.utils.game_status import GameStatus
from neural_network import *


class Player:
    # public:
    def __init__(self, nn : NN.NeuralNetwork, show_display: bool = False, move_time: float = 0):
        pygame.init()

        self.__nn = nn
        self.__move_time = move_time
        self.__show_display = show_display
        if show_display:
            self.__snake: SnakeHandler = SnakeHandler(True)
        else:
            self.__snake: RawSnake = RawSnake(False)

    def set_nn(self, nn : NN.NeuralNetwork):
        self.__nn = nn


    def play(self):
        if self.__show_display:
            time.sleep(1)

        self.__snake.new_game()
        steps = 0.

        while (gs := self.__snake.getGameStatus()) == GameStatus.GAME_ON:
            sensors = self.__snake.getSensors()

            inputVector = np.array(sensors.toNormalizedList())

            outputVector = self.__nn(inputVector)
            next_move = Direction(np.argmax(outputVector))
            self.__snake.make_move(next_move)

            steps += 1.

            if self.__show_display:
                time.sleep(self.__move_time)
        
        if isinstance(self.__snake, SnakeHandler):
            self.__snake.quit()
            return
        
        apples = self.__snake.getScore()

        return steps + (2**apples + 500 * apples**2.1) - ((apples**1.2) * (0.25 * steps)**1.3)

        return float(self.__snake.getScore())
    


def snake_fitness(nn : NN.NeuralNetwork, player : Player, lifes = 1):
    player.set_nn(nn)
    return sum(player.play() for _ in range(lifes) ) / lifes


if __name__ == "__main__":

    model = NN.NeuralNetwork()
    model.add_input_layer(32)
    model.add_hidden_layer(24, NN.relu)
    model.add_hidden_layer(12, NN.relu)
    model.add_output_layer(4, NN.softmax)


    player = Player(model)

    pop_size = 500
    init_pop = [model.copy() for _ in range(pop_size)]
    for i in range(pop_size):
        init_pop[i].g_rand()

    evolution_history, statistics = GA.genetic_algorithm(
        init_pop,
        (snake_fitness, player, 10),
        (GA.roulette_wheel_selection,),
        (NN.crossover,),
        [(NN.g_neuron_mutation, 0.6, 1),
        (NN.g_mutate, 0.05)],
        0.05, 1000)
    
    _, best_f, worst_f, avg_f, Q1_f, median_f, Q3_f, std_f = zip(*statistics)

    snake, _ = evolution_history[-1]

    snake.save("best_snake")
    

    plt.plot(best_f, label="best")
    plt.plot(avg_f, label="avg")
    plt.semilogy()
    plt.legend()
    plt.show()

    while True:
        player = Player(snake, True, 0.20)
        player.play()