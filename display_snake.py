import numpy as np
import pygame
import time
from snake.snake_handler import SnakeHandler
from snake.utils.direction import Direction
from snake.utils.game_status import GameStatus
from NN.neural_network import NeuralNetwork
from NN.activation_functions import *



class Player:
    def __init__(self, brain : NeuralNetwork, move_time: float = 0.20) -> None:

        self.__brain = brain
        self.__snake = SnakeHandler(True)
        self.__move_time = move_time

    def set_brain(self, brain : NeuralNetwork) -> None:
        self.__brain = brain

    
    def play(self):
        time.sleep(1)
        while True:
            while gs := self.__snake.getGameStatus() == GameStatus.GAME_ON:
                sensors = self.__snake.getSensors()

                inputVector = np.array(sensors.toNormalizedList())

                outputVector = self.__brain(inputVector)

                next_move = Direction(np.argmax(outputVector))
                self.__snake.make_move(next_move)

                time.sleep(self.__move_time)
            time.sleep(1)
            self.__snake.new_game()


if __name__ == "__main__":
    pygame.init()

    model = NeuralNetwork()
    model.add_input_layer(32)
    model.add_hidden_layer(24, relu)
    model.add_hidden_layer(12, relu)
    model.add_output_layer(4, softmax)

    path = "uniform_biases.npz"
    brain = model.copy()
    brain.load(path)

    player = Player(brain, move_time=0.05)

    player.play()
