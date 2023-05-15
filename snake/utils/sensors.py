from snake.utils.direction import Direction
from snake.utils.snake_config import GRID_HEIGHT, GRID_WIDTH

MAX_DISTANCE = max(GRID_WIDTH, GRID_HEIGHT)


class Sensors:

    class VisionLine:
        def __init__(self):
            self.distance: int = -1
            self.is_apple: bool = False
            self.is_snake_body: bool = False

        def __str__(self):
            return "distance: " + str(self.distance) + ", is_apple: " + str(self.is_apple) + ", is_snake_body: " + \
                   str(self.is_snake_body)

        def toNormalizedList(self):
            return [self.distance / MAX_DISTANCE, int(self.is_apple), int(self.is_snake_body)]

        def toNormalizedList2(self):
            return [1.0 / max(self.distance, 1), int(self.is_apple), int(self.is_snake_body)]

    def __init__(self):
        self.head_direction: Direction = Direction.RIGHT
        self.tail_direction: Direction = Direction.RIGHT
        self.visions: list[Sensors.VisionLine] = [self.VisionLine() for _ in range(8)]

    def __str__(self):
        s = "head direction: " + str(self.head_direction) + "\ntail direction: " + str(self.tail_direction)
        for vl in self.visions:
            s += "\n" + str(vl)
        return s

    # binary apple and body sensors, distance to walls normalized in our way (min:0.0, max:1.0)
    def toNormalizedList(self):
        return [value for vision in self.visions for value in vision.toNormalizedList()] + \
               self.head_direction.toBinaryList() + self.tail_direction.toBinaryList()

    # binary apple and body sensors, distance to walls normalized same as in youtube video (min:1.0, max:x->0+)
    def toNormalizedList2(self):
        return [value for vision in self.visions for value in vision.toNormalizedList2()] + \
               self.head_direction.toBinaryList() + self.tail_direction.toBinaryList()
