from numba import njit
import numpy as np
from numba import int16, boolean, void, int16, prange
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

@njit(boolean(int16, int16, UniTuple(int16, 2), int16, int16[:, :]), cache=True)
def check_line(row, col, direction, player, board):

    dr, dc = direction
    opponent = 2 if player == 1 else 1
    r, c = row + dr, col + dc
    opponent_tiles = 0      
        
    while (0 <= r < 8 and 0 <= c < 8) and board[r, c] == opponent:
        r, c = r + dr, c + dc
        opponent_tiles += 1
            
    if 0 <= r < 8 and 0 <= c < 8 and board[r, c] == player and opponent_tiles > 0:
        return True
            
    return False
    
@njit(boolean(UniTuple(int16, 2), int16, int16[:, :], int16[:, :]), cache = True)
def is_valid_move(move, player, board, directions):
        row, col = move
        if board[row, col] != 0:
            return False
        
        for dr, dc in directions:
            if check_line(row, col, (dr, dc), player, board):
                return True
        return False

@njit(int16[:, :](int16, int16[:, :]), cache=True)
def get_possible_moves(player, board):
    moves = np.zeros((50, 2), dtype=np.int16)  # max 60 possible moves on an 8x8 board
    move_count = 0
    directions = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=np.int16)
    
    for r in range(8):
        for c in range(8):
            if board[r, c] == 0 and is_valid_move((r, c), player, board, directions):
                moves[move_count] = (int16(r), int16(c))
                move_count += 1
                
    return moves[:move_count]

@njit(void(UniTuple(int16, 2), int16, int16[:, :]), cache = True)
def flip_tiles(move, player, board):
    r, c = move
    board[r, c] = player
    directions = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], dtype=np.int16)
    for dr, dc in directions:
        if check_line(r, c, (dr, dc), player, board):
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] != player:
                board[rr, cc] = player
                rr, cc = rr + dr, cc + dc