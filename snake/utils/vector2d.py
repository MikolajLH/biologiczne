
class Vector2d:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __add__(self, other):
        return Vector2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2d(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return (self.x == other.x and
                self.y == other.y)

    def __copy__(self):
        return Vector2d(self.x, self.y)

    def toList(self):
        return [self.x, self.y]