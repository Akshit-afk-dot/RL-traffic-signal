QUIT = "QUIT"


class Surface:
    def fill(self, color):
        return None


class _Draw:
    @staticmethod
    def rect(surface, color, rect):
        return None

    @staticmethod
    def circle(surface, color, center, radius):
        return None


draw = _Draw()


class _Display:
    def set_mode(self, size):
        return Surface()

    def set_caption(self, caption):
        return None

    def flip(self):
        return None


display = _Display()


class _Event:
    @staticmethod
    def get():
        return []


event = _Event()


class _Clock:
    def tick(self, fps):
        return None


class _Time:
    @staticmethod
    def get_ticks():
        return 0

    @staticmethod
    def Clock():
        return _Clock()


time = _Time()


def init():
    return None


def quit():
    return None
