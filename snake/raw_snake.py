import pygame
import random
from copy import copy

from snake.utils.game_status import GameStatus
from snake.utils.sensors import Sensors
from snake.utils.snake_config import *
from snake.utils.vector2d import Vector2d
from snake.utils.direction import Direction


vision_lines_vectors = [Vector2d(0, -1), Vector2d(1, -1), Vector2d(1, 0), Vector2d(1, 1),
                        Vector2d(0, 1), Vector2d(-1, 1), Vector2d(-1, 0), Vector2d(-1, -1)]


class RawSnake:
    # public:

    def __init__(self, show_display=False):
        pygame.init()

        self.__show_display = show_display
        if show_display:
            self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.__sensors: Sensors = Sensors()
        self.new_game()

    def new_game(self):
        self.__reset()
        self.__draw_screen()

    def make_move(self, direction) -> bool:
        if self.__game_status != GameStatus.GAME_ON:
            return False

        if direction == Direction.UP and self.__snake_head.direction != Direction.DOWN:
            self.__snake_head.direction = Direction.UP
        elif direction == Direction.DOWN and self.__snake_head.direction != Direction.UP:
            self.__snake_head.direction = Direction.DOWN
        elif direction == Direction.LEFT and self.__snake_head.direction != Direction.RIGHT:
            self.__snake_head.direction = Direction.LEFT
        elif direction == Direction.RIGHT and self.__snake_head.direction != Direction.LEFT:
            self.__snake_head.direction = Direction.RIGHT

        self.__move_snake()
        self.__update_game_status()
        self.__update_sensors()
        self.__draw_screen()

        return self.__game_status == GameStatus.GAME_ON

    def getGameStatus(self):
        return self.__game_status

    def getSensors(self):
        return self.__sensors

    def getScore(self):
        return len(self.__snake_body) - 3

    class SnakeCell:
        def __init__(self, position: Vector2d, direction: Direction):
            self.position: Vector2d = position
            self.direction: Direction = direction

        def __copy__(self):
            return RawSnake.SnakeCell(copy(self.position), self.direction)

    # private:

    def __reset(self):
        self.__snake_head = self.SnakeCell(Vector2d(GRID_WIDTH // 2, GRID_HEIGHT // 2), Direction.RIGHT)
        self.__snake_body = [self.SnakeCell(Vector2d(self.__snake_head.position.x + i, self.__snake_head.position.y),
                                            Direction.RIGHT) for i in range(0, -3, -1)]
        self.__food_position = self.__get_random_food_position()
        self.__game_status = GameStatus.GAME_ON
        self.__update_sensors()
        self.__no_food_moves_counter = 0

    def __move_snake(self):
        self.__snake_head.position += self.__snake_head.direction.toUnitVector()
        self.__snake_body.insert(0, copy(self.__snake_head))

        if self.__snake_head.position == self.__food_position:
            self.__food_position = self.__get_random_food_position()
            self.__no_food_moves_counter = 0
        else:
            self.__snake_body.pop()

    def __get_random_food_position(self) -> Vector2d:
        if len(self.__snake_body) == GRID_SIZE:
            return

        positions = [cell.position for cell in self.__snake_body]
        while True:
            position = Vector2d(random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if position not in positions:
                return position

    def __update_game_status(self):
        if len(self.__snake_body) == GRID_SIZE:
            self.__game_status = GameStatus.GAME_WON
            return

        self.__no_food_moves_counter += 1
        if self.__no_food_moves_counter >= MAX_NO_FOOD_MOVES:
            self.__game_status = GameStatus.GAME_OVER_STARVED
            return

        x, y = self.__snake_head.position.toList()
        self.__game_status = GameStatus.GAME_ON
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            self.__game_status = GameStatus.GAME_OVER_WALL
        elif self.__snake_head.position in [cell.position for cell in self.__snake_body[1:]]:
            self.__game_status = GameStatus.GAME_OVER_BODY

    def __update_sensors(self):
        self.__sensors.head_direction = self.__snake_head.direction
        self.__sensors.tail_direction = self.__snake_body[-1].direction
        body_positions = [cell.position for cell in self.__snake_body]
        for i, unit_vector in enumerate(vision_lines_vectors):
            v = self.__snake_head.position
            self.__sensors.visions[i].is_apple = False
            self.__sensors.visions[i].is_snake_body = False
            while v.x >= 0 and v.x < GRID_WIDTH and v.y >= 0 and v.y < GRID_HEIGHT:
                v += unit_vector
                if v == self.__food_position:
                    self.__sensors.visions[i].is_apple = True
                elif v in body_positions:
                    self.__sensors.visions[i].is_snake_body = True

            dx, dy = (v - self.__snake_head.position).toList()
            self.__sensors.visions[i].distance = max(abs(dx), abs(dy))

    # drawing

    def __draw_screen(self):
        if not self.__show_display:
            return
        self.__screen.fill(BG_COLOR)
        self.__draw_grid()
        self.__draw_snake()
        self.__draw_food()
        score_text = f'Score: {len(self.__snake_body) - 3}'
        pygame.display.set_caption(score_text)
        pygame.display.update()

    def __draw_grid(self):
        for x in range(0, SCREEN_WIDTH, SQUARE_SIZE):
            pygame.draw.line(self.__screen, (40, 40, 40), (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, SQUARE_SIZE):
            pygame.draw.line(self.__screen, (40, 40, 40), (0, y), (SCREEN_WIDTH, y))

    def __draw_snake(self):
        SNAKE_COLOR = ALIVE_SNAKE_COLOR if self.__game_status == GameStatus.GAME_ON else DEAD_SNAKE_COLOR
        for cell in self.__snake_body:
            pygame.draw.rect(self.__screen, SNAKE_COLOR,
                             pygame.Rect(cell.position.x * SQUARE_SIZE, cell.position.y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def __draw_food(self):
        if self.__food_position is None:
            return
        pygame.draw.rect(self.__screen, FOOD_COLOR,
                         pygame.Rect(self.__food_position.x * SQUARE_SIZE, self.__food_position.y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
