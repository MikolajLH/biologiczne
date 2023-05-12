from random import randint
import time
import pygame

from snake.snake_handler import SnakeHandler
from snake.raw_snake import RawSnake
from snake.utils.direction import Direction
from snake.utils.game_status import GameStatus
from neural_network import *


class Player:
    # public:
    def __init__(self, dna: DNA, show_display: bool = False, move_time: float = 0):
        pygame.init()

        self.__brain = SnakeBrain(dna)
        self.__move_time = move_time
        self.__show_display = show_display
        if show_display:
            self.__snake: SnakeHandler = SnakeHandler(True)
        else:
            self.__snake: RawSnake = RawSnake(False)

    def start(self):
        while True:
            if self.__show_display:
                time.sleep(1)

            while self.__snake.getGameStatus() == GameStatus.GAME_ON:

                sensors = self.__snake.getSensors()
                inputValues = sensors.toNormalizedList()

                self.__brain.SetInputValues(inputValues)
                # next_move = self.__brain.GetOutput()
                next_move = get_random_move()  # temporary

                self.__snake.make_move(next_move)

                print(inputValues)
                time.sleep(self.__move_time)

            self.__snake.new_game()

        # if isinstance(self.__snake, SnakeHandler):
        #     self.__snake.quit()


def get_random_move() -> Direction:
    rand_index = randint(0, 3)
    directions = [e for e in Direction]
    return directions[rand_index]


if __name__ == "__main__":
    player = Player(DNA(), True, 0.20)
    player.start()
