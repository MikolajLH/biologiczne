from snake.utils.direction import Direction


class Sensors:

    class VisionLine:
        def __init__(self):
            self.distance: int = 0
            self.is_apple: bool = False
            self.is_snake_body: bool = False

        def __str__(self):
            return "distance: " + str(self.distance) + ", is_apple: " + str(self.is_apple) + ", is_snake_body: " + \
                   str(self.is_snake_body)

    def __init__(self):
        self.head_direction: Direction = None
        self.tail_direction: Direction = None
        self.visions: list[Sensors.VisionLine] = [self.VisionLine() for _ in range(8)]

    def __str__(self):
        s = "head direction: " + str(self.head_direction) + "\ntail direction: " + str(self.tail_direction)
        for vl in self.visions:
            s += ("\n" + str(vl))
        return s
