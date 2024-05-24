from numba import njit
import numpy as np
from numba import int16, boolean, void, int16, uint64, prange
from numba.types import UniTuple

# Define colors
BLACK = (49, 49, 49)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 144, 103)

# Define Graphic Constants
SCREEN_WIDTH = SCREEN_HEIGHT = 800
CELL_SIZE = SCREEN_HEIGHT // 8
CELL_SCALLING = 0.8

# Constants
INT16_POSINF = 32767
INT16_NEGINF = -32767

DIRECTIONS = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=np.int16)

FREE_CELL = 0
PLAYER1 = 1
PLAYER2 = 2