from enum import Enum
from snake.utils.vector2d import Vector2d


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def toUnitVector(self):
        match self.value:
            case Direction.UP.value:
                return Vector2d(0, -1)
            case Direction.RIGHT.value:
                return Vector2d(1, 0)
            case Direction.DOWN.value:
                return Vector2d(0, 1)
            case Direction.LEFT.value:
                return Vector2d(-1, 0)

    def __str__(self):
        return str(self.name)

    def toBinaryList(self):
        return [1 if i == self.value else 0 for i in range(len(Direction))]
