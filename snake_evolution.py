import NN
import numpy as np
import GA
import pygame
import os
from snake.snake_handler import SnakeHandler
from snake.raw_snake import RawSnake
from snake.utils.direction import Direction
from snake.utils.game_status import GameStatus
import matplotlib.pyplot as plt


class HiddenPlayer:
    def __init__(self, brain : NN.NeuralNetwork) -> None:

        self.__brain = brain
        self.__snake = RawSnake(False)

    def set_brain(self, brain : NN.NeuralNetwork) -> None:
        self.__brain = brain

    
    def play(self) -> tuple[float,float,float]:

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

        return steps + (2**apples + 500 * apples**2.1) - ((apples**1.2) * (0.25 * steps)**1.3), apples, steps



def snake_fitness(nn : NN.NeuralNetwork, player : HiddenPlayer, lifes : float = 1):
    assert lifes > 0

    player.set_brain(nn)
    f, *extra = np.sum(np.array(player.play()) for _ in range(lifes) ) / lifes

    return f, *extra


if __name__ == "__main__":
    pygame.init()

    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

    model = NN.NeuralNetwork()
    model.add_input_layer(32)
    model.add_hidden_layer(24, NN.relu)
    model.add_hidden_layer(12, NN.relu)
    model.add_output_layer(4, NN.softmax)

    player = HiddenPlayer(model)

    pop_size = 500
    init_pop = [model.copy() for _ in range(pop_size)]

    for i in range(pop_size):
        init_pop[i].g_rand()

    evolution_history, statistics = GA.genetic_algorithm(
        init_pop,
        (snake_fitness, player),
        (GA.roulette_wheel_selection,),
        (NN.crossover,),
        [(NN.g_neuron_mutation, 0.6, 1),
        (NN.g_mutate, 0.05)],
        0.05, 1,
        extra_info_function= lambda a, s: f"avg_apples: {a}, avg_steps: {s}"
        )
        
    _, best_f, worst_f, avg_f, Q1_f, median_f, Q3_f, std_f = zip(*statistics)


    best_snake, _ = evolution_history[-1]

    best_snake.save("best_snake")

    plt.plot(best_f, label="best")
    plt.plot(avg_f, label="avg")
    plt.semilogy()
    plt.legend()
    plt.show()
