from enum import Enum


class GameStatus(Enum):
    GAME_ON = 0,
    GAME_OVER_WALL = 1,
    GAME_OVER_BODY = 2,
    GAME_OVER_STARVED = 3,
    GAME_OVER_QUIT = 4,     # Invalid ending of game
    GAME_WON = 5
