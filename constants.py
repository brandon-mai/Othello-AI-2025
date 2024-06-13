from numba import njit
import numpy as np
from numba import int16, boolean, void, int16, uint64, prange
from numba.types import UniTuple

# Colors
BLACK = (49, 49, 49)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 144, 103)
RED = (255, 0, 0)

# Graphic Constants
SCREEN_WIDTH = SCREEN_HEIGHT = 800
CELL_SIZE = SCREEN_HEIGHT // 8
CELL_SCALLING = 0.8

# Constants
INT16_POSINF = int16(32767)
INT16_NEGINF = int16(-32767)

PLAYER_1 = 1
PLAYER_2 = 2