import random

DEFAULT_COLOR = BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
LIME = (0, 255, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
SILVER = (192, 192, 192)
GRAY = (128, 128, 128)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)
NAVY = (0, 0, 128)


def get_random_color() -> tuple[int, int, int]:
    """Generate a random RGB color.
    Returns:
        tuple[int, int, int]: A tuple representing the RGB color.
    """
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
