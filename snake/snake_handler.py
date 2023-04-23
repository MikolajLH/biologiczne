import threading
import pygame

from snake.snake import Snake
from snake.utils.direction import Direction
from snake.utils.sensors import Sensors
from snake.utils.game_status import GameStatus


class SnakeHandler:
    class GameStatusWrapper:
        current_status = GameStatus.GAME_ON

    def __init__(self, show_display=False):
        self.snake = None
        self.__game_sensors = Sensors()
        self.__game_status_wrapper = self.GameStatusWrapper()
        self.__show_display = show_display
        self.__game_thread = threading.Thread(target=self.__run_game)
        self.__game_thread.start()

    def __run_game(self):
        self.snake = Snake(self.__game_status_wrapper, self.__game_sensors, self.__show_display)

    def quit(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.__game_thread.join()

    def getSensors(self) -> Sensors():
        return self.__game_sensors

    def getGameStatus(self):
        return self.__game_status_wrapper.current_status

    @staticmethod
    def make_move(direction: Direction):
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'move': direction}))

    @staticmethod
    def new_game():
        pygame.event.post(pygame.event.Event(pygame.SYSWMEVENT, msg="new_game"))
