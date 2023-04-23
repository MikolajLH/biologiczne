from random import randint
import time

from snake.snake_handler import SnakeHandler
from snake.raw_snake import RawSnake

from snake.utils.direction import Direction
from snake.utils.game_status import GameStatus


def get_random_move() -> Direction:
    rand_index = randint(0, 3)
    directions = [e for e in Direction]
    return directions[rand_index]


if __name__ == "__main__":
    # Sample usage and description of Snake classes:

    # SnakeHandler - second thread, display works good with small delay
    # No display and no delay may result in problems with synchronization (sensors don't work properly)
    snake_handler = SnakeHandler(True)
    for _ in range(2):
        while snake_handler.getGameStatus() is GameStatus.GAME_ON:
            move = get_random_move()
            SnakeHandler.make_move(move)
            print(snake_handler.getSensors())
            print(snake_handler.getGameStatus())
            time.sleep(0.1)
        snake_handler.new_game()
        time.sleep(2)
    snake_handler.quit()

    # RawSnake - same thread, good without display and for instant gameplay
    # With display it will say that window is not responding, because there isn't any main loop that waits for events
    snake = RawSnake(False)
    for _ in range(2):
        while snake.getGameStatus() == GameStatus.GAME_ON:
            move = get_random_move()
            snake.make_move(move)
            # print(snake.getSensors())
            print(snake.getGameStatus())
        snake.new_game()
