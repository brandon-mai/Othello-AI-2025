from numba import njit, uint64
import time
import cProfile


LeftMask = 0x7F7F7F7F7F7F7F7F
RightMask = 0xFEFEFEFEFEFEFEFE

@njit(uint64(uint64), cache = True, nogil = True)
def left(x):
    return (x >> 1) & LeftMask

@njit(uint64(uint64), cache = True, nogil = True)
def right(x):
    return (x << 1) & RightMask

@njit(uint64(uint64), cache = True, nogil = True)
def up(x):
    return x >> 8

@njit(uint64(uint64), cache = True, nogil = True)
def down(x):
    return x << 8

@njit(uint64(uint64), cache = True, nogil = True)
def up_left(x):
    return (x >> 9) & LeftMask

@njit(uint64(uint64), cache = True, nogil = True)
def up_right(x):
    return (x >> 7) & RightMask

@njit(uint64(uint64), cache = True, nogil = True)
def down_right(x):
    return (x << 9) & RightMask

@njit(uint64(uint64), cache = True, nogil = True)
def down_left(x):
    return (x << 7) & LeftMask

@njit(cache = True, nogil = True)
def validate_one_direction(shift, player_pieces, opponent_pieces, empty_squares):
    flood = shift(player_pieces)
    potential = flood & opponent_pieces
    valid_plays = 0

    potential = shift(potential)
    valid_plays |= potential & empty_squares
    potential = potential & opponent_pieces

    potential = shift(potential)
    valid_plays |= potential & empty_squares
    potential = potential & opponent_pieces

    potential = shift(potential)
    valid_plays |= potential & empty_squares
    potential = potential & opponent_pieces

    potential = shift(potential)
    valid_plays |= potential & empty_squares
    potential = potential & opponent_pieces

    potential = shift(potential)
    valid_plays |= potential & empty_squares
    potential = potential & opponent_pieces

    potential = shift(potential)
    valid_plays |= potential & empty_squares
    potential = potential & opponent_pieces

    return valid_plays

@njit(uint64(uint64, uint64), cache = True, nogil = True)
def possible_moves(player_pieces, opponent_pieces):
    
    empty_squares = ~(player_pieces | opponent_pieces)
    return  validate_one_direction(up, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(up_right, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(right, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(down_right, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(down, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(down_left, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(left, player_pieces, opponent_pieces, empty_squares) |\
            validate_one_direction(up_left, player_pieces, opponent_pieces, empty_squares)


def print_board(black_pieces, white_pieces):
        board = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(64):
            if (black_pieces >> i) & 1:
                board[i // 8][i % 8] = 'B'
            elif (white_pieces >> i) & 1:
                board[i // 8][i % 8] = 'W'
        
        print("  a b c d e f g h")
        for i, row in enumerate(board):
            print(f"{i+1} {' '.join(row)}")
            
def print_single_board(pieces):
        board = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(64):
            if (pieces >> i) & 1:
                board[i // 8][i % 8] = 'X'
        
        print("  a b c d e f g h")
        for i, row in enumerate(board):
            print(f"{i+1} {' '.join(row)}")
            
black_pieces = 0x0000000810000000
white_pieces = 0x0000001008000000

print_board(black_pieces, white_pieces)
print_single_board(possible_moves(black_pieces, white_pieces))

start = time.perf_counter()
for _ in range(10000):
    possible_moves(black_pieces, white_pieces)
tot = time.perf_counter()-start
print(tot/10000)